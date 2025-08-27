import math
import torch
import torch.nn.functional as F


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



