import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        compact=True,
        residual=True,
        circular_padding=False,
        cat=True,
    ):
        super().__init__()
        self.name = "unet"
        self.compact = compact
        self.residual = residual
        self.cat = cat

        self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(
            ch_in=in_channels, ch_out=64, circular_padding=circular_padding
        )
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConv(ch_in=1024, ch_out=512)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConv(ch_in=512, ch_out=256)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up3 = UpConv(ch_in=256, ch_out=128)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.up2 = UpConv(ch_in=128, ch_out=64)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_final = nn.Conv3d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoding path
        cat_dim = 1
        x1 = self.conv1(x)

        x2 = self.pooling(x1)
        x2 = self.conv2(x2)

        x3 = self.pooling(x2)
        x3 = self.conv3(x3)

        x4 = self.pooling(x3)
        x4 = self.conv4(x4)

        if not self.compact:
            x5 = self.pooling(x4)
            x5 = self.conv5(x5)

            d = self.up5(x5)
            if self.cat:
                d = torch.cat((x4, d), dim=cat_dim)
                d = self.up_conv5(d)

        # decoding + concat path
        d = self.up4(x4)
        if self.cat:
            d = torch.cat((x3, d), dim=cat_dim)
            d = self.up_conv4(d)

        d = self.up3(d)
        if self.cat:
            d = torch.cat((x2, d), dim=cat_dim)
            d = self.up_conv3(d)

        d = self.up2(d)
        if self.cat:
            d = torch.cat((x1, d), dim=cat_dim)
            d = self.up_conv2(d)

        d = self.conv_final(d)

        return d + x if self.residual else d


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, circular_padding=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                ch_in,
                ch_out,
                kernel_size=3,
                padding=1,
                padding_mode="circular" if circular_padding else "zeros",
            ),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
