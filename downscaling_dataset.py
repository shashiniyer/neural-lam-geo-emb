# Standard library
import datetime as dt
import glob
import os

# Third-party
import numpy as np
import torch
import torch.nn.functional as F

# First-party
from neural_lam import constants, utils


class DownscalingDataset(torch.utils.data.Dataset):
    """
    For our dataset:
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    dim_x = 268
    dim_y = 238
    N_grid = 268x238 = 63784
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    """

    def __init__(
        self,
        dataset_name,
        pred_length=1,
        split="train",
        subsample_step=3,
        standardize=True,
        subset=False,
        control_only=False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        data_dir_path = "/proj/berzelius-2022-164/weather/neural_lam_datasets"
        dataset_dir_path = os.path.join(data_dir_path, dataset_name)
        self.sample_dir_path = os.path.join(
            dataset_dir_path, "samples", split
        )

        member_file_regexp = (
            "nwp*mbr000.npy" if control_only else "nwp*mbr*.npy"
        )
        sample_paths = glob.glob(
            os.path.join(self.sample_dir_path, member_file_regexp)
        )
        self.sample_names = [path.split("/")[-1][4:-4] for path in sample_paths]
        # Now on form "yyymmddhh_mbrXXX"

        if subset:
            self.sample_names = self.sample_names[:1]  # Limit to 50 samples, changed to 1 sample for testing

        self.sample_length = pred_length # 2 init states # Changed sample length to 1 since we only want to sample the current step.
        self.subsample_step = subsample_step
        self.original_sample_length = (
            65 // self.subsample_step
        )  # 21 for 3h steps
        assert (
            self.sample_length <= self.original_sample_length
        ), "Requesting too long time series samples"

        # Set up for standardization
        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, data_dir_path, "cpu")
            self.data_mean, self.data_std, self.flux_mean, self.flux_std = (
                ds_stats["data_mean"],
                ds_stats["data_std"],
                ds_stats["flux_mean"],
                ds_stats["flux_std"],
            )

        # If subsample index should be sampled (only duing training)
        self.random_subsample = split == "train"

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        # === Sample ===
        sample_name = self.sample_names[idx]
        sample_path = os.path.join(
            self.sample_dir_path, f"nwp_{sample_name}.npy"
        )
        try:
            full_sample = torch.tensor(
                np.load(sample_path), dtype=torch.float32
            )  # (N_t', dim_x, dim_y, d_features')
        except ValueError:
            print(f"Failed to load {sample_path}")

        # Only use every ss_step:th time step, sample which of ss_step
        # possible such time series
        if self.random_subsample:
            subsample_index = torch.randint(0, self.subsample_step, ()).item()
        else:
            subsample_index = 0
        subsample_end_index = self.original_sample_length * self.subsample_step
        sample = full_sample[
            subsample_index : subsample_end_index : self.subsample_step
        ]

        # (N_t, dim_x, dim_y, d_features')

        # Remove feature 15, "z_height_above_ground"
        sample = torch.cat(
            (sample[:, :, :, :15], sample[:, :, :, 16:]), dim=3
        )  # (N_t, dim_x, dim_y, d_features)

        # Accumulate solar radiation instead of just subsampling
        rad_features = full_sample[:, :, :, 2:4]  # (N_t', dim_x, dim_y, 2)
        # Accumulate for first time step
        init_accum_rad = torch.sum(
            rad_features[: (subsample_index + 1)], dim=0, keepdim=True
        )  # (1, dim_x, dim_y, 2)
        # Accumulate for rest of subsampled sequence
        in_subsample_len = (
            subsample_end_index - self.subsample_step + subsample_index + 1
        )
        rad_features_in_subsample = rad_features[
            (subsample_index + 1) : in_subsample_len
        ]  # (N_t*, dim_x, dim_y, 2), N_t* = (N_t-1)*ss_step
        _, dim_x, dim_y, _ = sample.shape
        rest_accum_rad = torch.sum(
            rad_features_in_subsample.view(
                self.original_sample_length - 1,
                self.subsample_step,
                dim_x,
                dim_y,
                2,
            ),
            dim=1,
        )  # (N_t-1, dim_x, dim_y, 2)
        accum_rad = torch.cat(
            (init_accum_rad, rest_accum_rad), dim=0
        )  # (N_t, dim_x, dim_y, 2)
        # Replace in sample
        sample[:, :, :, 2:4] = accum_rad

        # Flatten spatial dim
        # sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features) # Removed flattening since we want to keep spatial dimensions

        # Uniformly sample time id to start sample from
        init_id = torch.randint(
            0, 1 + self.original_sample_length - self.sample_length, ()
        )

        sample = sample[init_id : (init_id + self.sample_length)]  # (sample_length, dim_x, dim_y, d_features)
        # (sample_length, N_grid, d_features)

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std

        # Split up sample in init. states and target states
        # init_states = sample[:2]  # (2, N_grid, d_features)
        # target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        # === Forcing features ===
        # Now batch-static features are just part of forcing,
        # repeated over temporal dimension
        # Load water coverage
        # sample_datetime = sample_name[:10]
        # water_path = os.path.join(
        #     self.sample_dir_path, f"wtr_{sample_datetime}.npy"
        # )
        # water_cover_features = torch.tensor(
        #     np.load(water_path), dtype=torch.float32
        # ).unsqueeze(
        #     -1
        # )  # (dim_x, dim_y, 1)
        # # Flatten
        # water_cover_features = water_cover_features.flatten(0, 1)  # (N_grid, 1)
        # # Expand over temporal dimension
        # water_cover_expanded = water_cover_features.unsqueeze(0).expand(
        #     self.sample_length - 2, -1, -1  # -2 as added on after windowing
        # )  # (sample_len, N_grid, 1)

        # # TOA flux
        # flux_path = os.path.join(
        #     self.sample_dir_path,
        #     f"nwp_toa_downwelling_shortwave_flux_{sample_datetime}.npy",
        # )
        # flux = torch.tensor(np.load(flux_path), dtype=torch.float32).unsqueeze(
        #     -1
        # )  # (N_t', dim_x, dim_y, 1)

        # if self.standardize:
        #     flux = (flux - self.flux_mean) / self.flux_std

        # # Flatten and subsample flux forcing
        # flux = flux.flatten(1, 2)  # (N_t, N_grid, 1)
        # flux = flux[subsample_index :: self.subsample_step]  # (N_t, N_grid, 1)
        # flux = flux[
        #     init_id : (init_id + self.sample_length)
        # ]  # (sample_len, N_grid, 1)

        # # Time of day and year
        # dt_obj = dt.datetime.strptime(sample_datetime, "%Y%m%d%H")
        # dt_obj = dt_obj + dt.timedelta(
        #     hours=2 + subsample_index
        # )  # Offset for first index
        # # Extract for initial step
        # init_hour_in_day = dt_obj.hour
        # start_of_year = dt.datetime(dt_obj.year, 1, 1)
        # init_seconds_into_year = (dt_obj - start_of_year).total_seconds()

        # # Add increments for all steps
        # hour_inc = (
        #     torch.arange(self.sample_length) * self.subsample_step
        # )  # (sample_len,)
        # hour_of_day = (
        #     init_hour_in_day + hour_inc
        # )  # (sample_len,), Can be > 24 but ok
        # second_into_year = (
        #     init_seconds_into_year + hour_inc * 3600
        # )  # (sample_len,)
        # # can roll over to next year, ok because periodicity

        # # Encode as sin/cos
        # hour_angle = (hour_of_day / 12) * torch.pi  # (sample_len,)
        # year_angle = (
        #     (second_into_year / constants.SECONDS_IN_YEAR) * 2 * torch.pi
        # )  # (sample_len,)
        # datetime_forcing = torch.stack(
        #     (
        #         torch.sin(hour_angle),
        #         torch.cos(hour_angle),
        #         torch.sin(year_angle),
        #         torch.cos(year_angle),
        #     ),
        #     dim=1,
        # )  # (N_t, 4)
        # datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]
        # datetime_forcing = datetime_forcing.unsqueeze(1).expand(
        #     -1, flux.shape[1], -1
        # )  # (sample_len, N_grid, 4)

        # # Put forcing features together
        # forcing_features = torch.cat(
        #     (flux, datetime_forcing), dim=-1
        # )  # (sample_len, N_grid, d_forcing)

        # # Combine forcing over each window of 3 time steps
        # forcing_windowed = torch.cat(
        #     (
        #         forcing_features[:-2],
        #         forcing_features[1:-1],
        #         forcing_features[2:],
        #     ),
        #     dim=2,
        # )  # (sample_len-2, N_grid, 3*d_forcing)
        # # Now index 0 of ^ corresponds to forcing at index 0-2 of sample

        # # batch-static water cover is added after windowing,
        # # as it is static over time
        # forcing = torch.cat((water_cover_expanded, forcing_windowed), dim=2)
        # (sample_len-2, N_grid, forcing_dim)

        # Create high and low resolution states
        GT = sample.permute(0, 3, 1, 2)  # (sample_length, d_features, dim_x, dim_y)

        scale = 20
        mode="bilinear"
        LQ = F.interpolate(F.interpolate(GT, scale_factor=1/scale, mode=mode), size=GT.shape[-2:], mode=mode)  # Downsample and then upsample to get LQ

        # return init_states, target_states, forcing
        states = {
            "LQ": LQ.squeeze(0),
            "HQ": GT.squeeze(0),
        }

        return states


if __name__ == '__main__':
    # Load data
    train_loader = torch.utils.data.DataLoader(
        DownscalingDataset(
            "meps",
            pred_length=1,
            split="train",
            subsample_step=3,
            subset=True,
            control_only=False,
        ),
        4,
        shuffle=True,
        num_workers=4,
    )

    # Get one batch from train_loader
    for states in train_loader:
        print(states["LQ"].shape)
        print(states["HQ"].shape)
        break  # Stop after first batch
