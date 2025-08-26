import os
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from scipy import signal
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import login, snapshot_download
from tqdm.auto import tqdm
import neptune


    

def cuda_to_np(tensor):
    return tensor.cpu().detach().numpy()


def get_stats(tensor, p=True, r=False, minmax=False):
    if minmax:
        min, max = tensor.min(), tensor.max()
        mean, std = tensor.mean(), tensor.std()
        if p: print(f"Min: {min}, Max: {max} ,Mean: {mean}, Std: {std}")
        if r: return min, max, mean, std
    else:
        mean, std = tensor.mean(), tensor.std()
        if p: print(f"Mean: {mean}, Std: {std}")
        if r: return mean, std
    
    
def zscore(tensor, mean=None, std=None):
    if mean is None: mean = tensor.mean()
    if std is None: std = tensor.std()
    return (tensor - mean) / (std + 1e-8)


def get_model_size(model):
    print(sum(p.numel() for p in model.parameters()) / 1e6)
    

def get_index(iterable):
    return random.randint(0, len(iterable) - 1)


def get_indices(iterable, n):
    return random.sample(range(len(iterable)), n)


def split(inputs, targets, seed):
    return train_test_split(
        inputs,
        targets, 
        test_size=0.2,
        shuffle=True, 
        random_state=seed
    ) 


def show_waves(waves, dpi=100):
    """
    waves: numpy array of shape (3, N)
    Creates three separate figures that stretch wide.
    """
    N = waves.shape[1]
    t = np.arange(N)

    # Wide aspect ratio; height modest so each window fills width
    for i in range(waves.shape[0]):
        fig = plt.figure(figsize=(14, 4), dpi=dpi)  # wide figure
        ax = fig.add_subplot(111)
        ax.plot(t, waves[i], linewidth=1)
        ax.set_title(f"Wave {i+1}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        fig.tight_layout()  # reduce margins to use width
        
    plt.show()
    
    
def hf_ds_download(hf_token, repo_id):
    login(hf_token[1:])
    return snapshot_download(repo_id, repo_type="dataset")


def get_spectra_features(X, b=False):
    """Create multi-channel features from spectra: raw, 1st derivative, 2nd derivative."""
    X_processed = np.zeros_like(X)
    # Baseline correction and SNV
    for i in tqdm(range(X.shape[0])):
        poly = np.polyfit(np.arange(X.shape[1]), X[i], 3)
        baseline = np.polyval(poly, np.arange(X.shape[1]))
        corrected_spec = X[i] - baseline
        #X_processed[i] = (corrected_spec - corrected_spec.mean()) / (corrected_spec.std() + 1e-8)
        X_processed[i] = corrected_spec
        
    # Calculate derivatives
    deriv1 = signal.savgol_filter(X_processed, window_length=11, polyorder=3, deriv=1, axis=1)
    deriv2 = signal.savgol_filter(X_processed, window_length=11, polyorder=3, deriv=2, axis=1)

    if b: return np.stack([X_processed, deriv1, deriv2], axis=1)
    return np.stack([deriv1, deriv2], axis=1)


def load_transfer_data():
    csv_path = os.path.join(path, files[5])
    df = pd.read_csv(csv_path)

    input_cols = df.columns[1:2049]
    target_cols = df.columns[2050:]

    targets  = df[target_cols].dropna().to_numpy()

    df = df[input_cols]
    df['Unnamed: 1'] = df['Unnamed: 1'].str.replace("[\[\]]", "", regex=True).astype('int64')
    df['Unnamed: 2048'] = df['Unnamed: 2048'].str.replace("[\[\]]", "", regex=True).astype('int64')

    inputs = df.to_numpy().reshape(-1, 2, 2048)
    inputs = inputs.mean(axis=1)
    
    return inputs, targets


def preprocess_transfer_data():
    inputs, targets = load_transfer_data()
    
    spectra_selection = np.logical_and(
        300 <= np.array([float(one) for one in range(2048)]),
        np.array([float(one) for one in range(2048)]) <= 1942,
    )
    
    inputs = inputs[:, spectra_selection]
    
    wns = np.array([
        float(one) for one in range(2048)
    ])[spectra_selection]
    wavenumbers = np.arange(300, 1943)
    
    interpolated_data = np.array(
        [np.interp(wavenumbers, xp=wns, fp=i) for i in inputs]
    )
    
    normed_spectra = interpolated_data / np.max(interpolated_data)
    return normed_spectra, targets


lower_bounds = {
    'anton_532': 200,
    'anton_785': 100,
    'kaiser': -37,
    'mettler_toledo': 300,
    'metrohm': 200,
    'tec5': 85,
    'timegate': 200,
    'tornado': 300,
}


upper_bounds = {
    'anton_532': 3500,
    'anton_785': 2300,
    'kaiser': 1942,
    'mettler_toledo': 3350,
    'metrohm': 3350,
    'tec5': 3210,
    'timegate': 2000,
    'tornado': 3300,
}


def get_data(path, name, lower=-1000, upper=10000):
    df = pd.read_csv(os.path.join(path, name))

    lower = max(lower, lower_bounds[name[:-4]])
    upper = min(upper, upper_bounds[name[:-4]])

    spectra_selection = np.logical_and(
        lower <= np.array([float(one) for one in df.columns[:-5]]),
        np.array([float(one) for one in df.columns[:-5]]) <= upper,
    )
    
    spectra = df.iloc[:, :-5].iloc[:, spectra_selection].values
    label = df.iloc[:, -5:-2].values

    wavenumbers = np.array([
        float(one) for one in df.columns[:-5]
    ])[spectra_selection]

    #indices = get_indices(spectra, num_samples)                         
    return spectra, label, wavenumbers


def load_datasets(path, ds_names, lower=-1000, upper=10000):
        
    lower = max(
        lower,
        *[lower_bounds[n[:-4]] for n in ds_names])
    
    upper = min(
        upper,
        *[upper_bounds[n[:-4]] for n in ds_names]
    )

    datasets = [get_data(path, name, lower, upper) for name in ds_names]
    wavenumbers = np.arange(lower, upper + 1)

    interpolated_data = [
        np.array([
            np.interp(
                wavenumbers,
                xp=wns,
                fp=spectrum,
            )
            for spectrum in spectra
        ])
        for spectra, _, wns in datasets
    ]

    normed_spectra = np.concatenate(
        [
            spectra / np.max(spectra)
            for spectra in interpolated_data
        ],
        axis=0,
    )

    labels = np.concatenate([ds[1] for ds in datasets])
    return normed_spectra, labels

    


  



import math


class Identity(torch.torch.nn.Module):
    def forward(self, x):
        return x


# this is not a resnet yet
class ReZeroBlock(torch.torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation_function,
        kernel_size,
        stride,
        dtype,
        norm_layer=None,
    ):
        super(ReZeroBlock, self).__init__()
        if norm_layer is None:
            norm_layer = torch.torch.nn.BatchNorm1d

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = divmod(kernel_size, 2)[0] if stride == 1 else 0

        # does not change spatial dimension
        self.conv1 = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            dtype=dtype,
        )
        self.bn1 = norm_layer(out_channels, dtype=dtype)
        # Both self.conv2 and self.downsample layers
        # downsample the input when stride != 1
        self.conv2 = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=out_channels,
            bias=False,
            dtype=dtype,
            padding=self.padding,
        )
        if stride > 1:
            down_conv = torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                dtype=dtype,
                # groups=out_channels,
            )
        else:
            down_conv = Identity()

        self.down_sample = torch.nn.Sequential(
            down_conv,
            norm_layer(out_channels),
        )
        self.bn2 = norm_layer(out_channels, dtype=dtype)
        # does not change the spatial dimension
        self.conv3 = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            dtype=dtype,
        )
        self.bn3 = norm_layer(out_channels, dtype=dtype)
        self.activation = activation_function(inplace=True)
        self.factor = torch.torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=dtype))

    def next_spatial_dim(self, last_spatial_dim):
        return math.floor(
            (last_spatial_dim + 2 * self.padding - self.kernel_size)
            / self.stride + 1
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # not really the identity, but kind of
        identity = self.down_sample(x)

        return self.activation(out * self.factor + identity)


class ResNetEncoder(torch.torch.nn.Module):
    def __init__(
        self,
        spectrum_size,
        cnn_encoder_channel_dims,
        activation_function,
        kernel_size,
        stride,
        dtype,
        num_blocks,
        verbose=False,
    ):
        super(ResNetEncoder, self).__init__()

        self.spatial_dims = [spectrum_size]
        layers = []
        for in_channels, out_channels in zip(
            cnn_encoder_channel_dims[:-1],
            cnn_encoder_channel_dims[1:],
        ):
            block = ReZeroBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                activation_function=activation_function,
                kernel_size=kernel_size,
                stride=stride,
                dtype=dtype,
            )
            layers.append(block)
            self.spatial_dims.append(block.next_spatial_dim(self.spatial_dims[-1]))
            for _ in range(num_blocks - 1):
                block = ReZeroBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    activation_function=activation_function,
                    kernel_size=kernel_size,
                    stride=1,
                    dtype=dtype,
                )
                layers.append(block)
                self.spatial_dims.append(block.next_spatial_dim(self.spatial_dims[-1]))

        self.resnet_layers = torch.torch.nn.Sequential(*layers)
        if verbose:
            print("CNN Encoder Channel Dims: %s" % (cnn_encoder_channel_dims))
            print("CNN Encoder Spatial Dims: %s" % (self.spatial_dims))

    def forward(self, x):
        return self.resnet_layers(x)


class ReZeroNet(torch.nn.Module):
    def __init__(
        self,
        spectra_channels,
        spectra_size,
        initial_cnn_channels,
        cnn_channel_factor,
        num_cnn_layers,
        kernel_size,
        stride,
        activation_function,
        fc_dims,
        fc_dropout=0.0,
        dtype=None,
        verbose=False,
        fc_output_channels=1,
        num_blocks=1,
        **kwargs,
    ):
        super().__init__()
        self.fc_output_channels = fc_output_channels
        self.dtype = dtype or torch.float32

        activation_function = getattr(torch.nn, activation_function)

        # Setup CNN Encoder
        cnn_encoder_channel_dims = [spectra_channels] + [
            int(initial_cnn_channels * (cnn_channel_factor**idx))
            for idx in range(num_cnn_layers)
        ]
        self.cnn_encoder = ResNetEncoder(
            spectrum_size=spectra_size,
            cnn_encoder_channel_dims=cnn_encoder_channel_dims,
            activation_function=activation_function,
            kernel_size=kernel_size,
            stride=stride,
            num_blocks=num_blocks,
            dtype=dtype,
            verbose=verbose,
        )
        self.fc_dims = [
            int(
                self.cnn_encoder.spatial_dims[-1]
            ) * int(cnn_encoder_channel_dims[-1])
        ] + fc_dims

        if verbose:
            print("Fc Dims: %s" % self.fc_dims)
        fc_layers = []
        for idx, (in_dim, out_dim) in enumerate(
                zip(self.fc_dims[:-2], self.fc_dims[1:-1])
        ):
            fc_layers.append(torch.nn.Linear(in_dim, out_dim))
            fc_layers.append(torch.nn.ELU())
            fc_layers.append(torch.nn.Dropout(fc_dropout / (2 ** idx)))
        fc_layers.append(
            torch.nn.Linear(
                self.fc_dims[-2],
                self.fc_dims[-1] * self.fc_output_channels,
            ),
        )
        self.fc_net = torch.nn.Sequential(*fc_layers)
        if verbose:
            num_params = sum(p.numel() for p in self.parameters())
            print("Number of Parameters: %s" % num_params)

    def forward(self, spectra):
        embeddings = self.cnn_encoder(spectra)
        forecast = self.fc_net(embeddings.view(-1, self.fc_dims[0]))
        if self.fc_output_channels > 1:
            forecast = forecast.reshape(
                -1, self.fc_output_channels, self.fc_dims[-1]
            )
        return forecast



