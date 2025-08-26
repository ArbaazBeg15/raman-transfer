import numpy as np
import random
import torch
from torch.utils.data import Dataset
import scipy.optimize


np_dtype_from_torch = {
    torch.float32: np.float32,
    torch.float64: np.float64,
}

class SpectralDataset(Dataset):
    def __init__(
        self,
        spectra,
        concentrations,
        dtype=None,
        spectra_mean_std=None,
        concentration_mean_std=None,
        combine_spectra_range=0.0,
        baseline_factor_bound=0.0,
        baseline_period_lower_bound=100.0,
        baseline_period_upper_bound=200.0,
        augment_slope_std=0.0,
        augment_intersept_std=0.0,
        rolling_bound=0,
        spectrum_rolling_sigma=0.0,
        augmentation_weight=0.1,
        original_datapoint_weight=1.,
    ):
        self.dtype = dtype or torch.float32
        self.combine_spectra_range = combine_spectra_range
        self.baseline_factor_bound = baseline_factor_bound
        self.augment_slope_std = augment_slope_std
        self.augment_intercept_std = augment_intersept_std
        self.baseline_period_lower_bound = baseline_period_lower_bound
        self.baseline_period_upper_bound = baseline_period_upper_bound
        self.rolling_bound = rolling_bound
        self.spectrum_rolling_sigma = spectrum_rolling_sigma
        self.augmentation_weight = torch.tensor(augmentation_weight, dtype=dtype)
        self.original_dp_weight = original_datapoint_weight

        # normalize spectra
        spectra = torch.tensor(spectra, dtype=dtype)

        if spectra_mean_std is None:
            self.s_mean = torch.mean(spectra)
            self.s_std = torch.std(spectra)
        else:
            self.s_mean, self.s_std = spectra_mean_std

        self.spectra = torch.divide(
            torch.subtract(spectra, self.s_mean),
            self.s_std,
        )

        self.dummy_wns = np.tile(
            np.arange(
                0., 1., 1. / self.spectra.shape[2],
                dtype=np_dtype_from_torch[self.dtype]
            )[None, :self.spectra.shape[2]],
            (self.spectra.shape[1], 1),
        )

        # normalize concentrations
        concentrations = torch.tensor(concentrations, dtype=dtype)
        if concentration_mean_std is None:
            self.concentration_means = torch.nanmean(concentrations, dim=0)

            self.concentration_stds = torch.maximum(
                torch.tensor(
                    [
                        torch.std(col[torch.logical_not(torch.isnan(col))])
                        for col in concentrations.T
                    ]
                ),
                torch.tensor([1e-3] * concentrations.shape[1]),
            )
        else:
            self.concentration_means = concentration_mean_std[0]
            self.concentration_stds = concentration_mean_std[1]

        self.concentrations = torch.divide(
            torch.subtract(
                concentrations,
                self.concentration_means,
            ),
            self.concentration_stds,
        )

    def pick_two(self, max_idx=None):
        max_idx = max_idx or len(self)
        return random.choices(range(max_idx), k=2)

    def __len__(self):
        return len(self.concentrations)

    def augment_spectra(self, spectra):
        if self.augment_slope_std > 0.0:

            def spectrum_approximation(x, slope, intercept):
                return (slope * x + intercept).reshape(-1, 1)[:, 0]

            slope, inter = scipy.optimize.curve_fit(
                spectrum_approximation,
                self.dummy_wns,
                spectra.reshape(-1, 1)[:, 0],
                p0=np.random.rand(2),
            )[0]

            new_slope = slope * (
                    np.random.gamma(
                        shape=1. / self.augment_slope_std,
                        scale=self.augment_slope_std,
                        size=1,
                    )
            )[0]
            new_intercept = inter * (
                1.0 + np.random.randn(1) * self.augment_intercept_std
            )[0]
            spectra += torch.tensor(
                (new_slope - slope)
            ) * self.dummy_wns + new_intercept - inter

        factor = self.baseline_factor_bound * torch.rand(size=(1,))
        offset = torch.rand(size=(1,)) * 2.0 * torch.pi
        period = self.baseline_period_lower_bound + (
            self.baseline_period_upper_bound - self.baseline_period_lower_bound
        ) * torch.rand(size=(1,))
        permutations = factor * torch.cos(
            2.0 * torch.pi / period * self.dummy_wns + offset
        )
        return self.roll_spectrum(
            spectra + permutations * spectra,
            delta=random.randint(-self.rolling_bound, self.rolling_bound),
        )

    def roll_spectrum(self, spectra, delta):
        num_spectra = spectra.shape[0]
        rolled_spectra = np.roll(spectra, delta, axis=1)
        if delta > 0:
            rolled_spectra[:, :delta] = (
                np.random.rand(num_spectra, delta) * self.spectrum_rolling_sigma + 1
            ) * rolled_spectra[:, delta:(delta + 1)]
        elif delta < 0:
            rolled_spectra[:, delta:] = (
                np.random.rand(num_spectra, -delta) * self.spectrum_rolling_sigma + 1
            ) * rolled_spectra[:, delta - 1:delta]
        return rolled_spectra

    def combine_k_items(self, indices, weights):
        return (
            # spectra
            torch.sum(
                torch.mul(weights[:, None, None], self.spectra[indices, :, :]),
                dim=0,
            ),
            # concentrations
            torch.sum(
                torch.mul(weights[:, None], self.concentrations[indices, :]),
                dim=0,
            )
        )

    def __getitem__(self, idx):
        if self.combine_spectra_range < 1e-12:
            spectrum = self.spectra[idx]
            spectrum = self.augment_spectra(spectrum)
            return {
                "spectra": spectrum,
                "concentrations": self.concentrations[idx],
                "label_weight": torch.tensor(1.0, dtype=self.dtype),
            }
        else:
            if random.random() < self.original_dp_weight:
                one_weight = 1.
                label_weight = torch.tensor(1.0, dtype=self.dtype)
            else:
                one_weight = random.uniform(0.0, self.combine_spectra_range)
                label_weight = self.augmentation_weight
            weights = torch.tensor([one_weight, (1 - one_weight)])
            # just pick two random indices
            indices = random.choices(range(len(self)), k=2)

            mixed_spectra, mixed_concentrations = self.combine_k_items(
                indices=indices,
                weights=weights,
            )
            mixed_spectra = self.augment_spectra(mixed_spectra)
            return mixed_spectra, mixed_concentrations, label_weight


def get_ds(inputs, targets, config, inputs_mean_std=None, targets_mean_std=None):
    return SpectralDataset(
        spectra=inputs[:, None, :],
        concentrations=targets,
        dtype=torch.float32,
        spectra_mean_std=inputs_mean_std,
        concentration_mean_std=targets_mean_std,
        combine_spectra_range=1.0,
        baseline_factor_bound=config["baseline_factor_bound"],
        baseline_period_lower_bound=config["baseline_period_lower_bound"],
        baseline_period_upper_bound=(config["baseline_period_lower_bound"] + config["baseline_period_span"]),
        augment_slope_std=config["augment_slope_std"],
        augment_intersept_std=0.0,
        rolling_bound=config["rolling_bound"],
        spectrum_rolling_sigma=0.01,
        augmentation_weight=0.1,
        original_datapoint_weight=1.,
    )