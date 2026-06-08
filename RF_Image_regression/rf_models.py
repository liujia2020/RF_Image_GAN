import inspect

import torch
import torch.nn as nn


MODEL_NAME_TO_CLASS = {
    "tiny": "TinyResidualRFNet",
    "tiny_residual": "TinyResidualRFNet",
    "tinyresidualrfnet": "TinyResidualRFNet",
    "tiny_baseline_conditioned": "TinyBaselineConditionedRFNet",
    "tiny_baseline": "TinyBaselineConditionedRFNet",
    "baseline_conditioned": "TinyBaselineConditionedRFNet",
    "grouped": "GroupedRFResidualNet",
    "grouped_rf": "GroupedRFResidualNet",
    "groupedrfresidualnet": "GroupedRFResidualNet",
    "earlymix": "EarlyMixWideResidualNet",
    "earlymix_wide": "EarlyMixWideResidualNet",
    "earlymixwideresidualnet": "EarlyMixWideResidualNet",
    "unet": "UNetResidualRFNet",
    "unet_residual": "UNetResidualRFNet",
    "unetresidualrfnet": "UNetResidualRFNet",
    "wide": "WideDeepResidualRFNet",
    "wide_deep": "WideDeepResidualRFNet",
    "wide_deep_residual": "WideDeepResidualRFNet",
    "widedeepresidualrfnet": "WideDeepResidualRFNet",
}


class TinyBaselineConditionedRFNet(nn.Module):
    """
    Tiny residual model with explicit baseline conditioning.

    Input:
        x        [B,1536,Z,X,Y]
        baseline [B,2,Z,X,Y]

    Internal input:
        concat([x, baseline], dim=1)
        [B,1538,Z,X,Y]

    Output:
        pred     [B,2,Z,X,Y]

    The network predicts:
        pred = baseline + residual
    """

    def __init__(self, in_channels=1536, hidden=64, out_channels=2):
        super().__init__()

        cond_in_channels = in_channels + out_channels

        self.net = nn.Sequential(
            nn.Conv3d(cond_in_channels, hidden, kernel_size=1, padding=0),
            nn.BatchNorm3d(hidden, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(32, out_channels, kernel_size=1, padding=0),
        )

        # Start close to baseline.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x, baseline):
        x_cond = torch.cat([x, baseline], dim=1)
        residual = self.net(x_cond)
        pred = baseline + residual
        return pred

class TinyResidualRFNet(nn.Module):
    """
    Baseline residual learned beamforming model.

    Input:
        x        [B,1536,Z,X,Y]
        baseline [B,2,Z,X,Y]

    Output:
        pred     [B,2,Z,X,Y]

    The network predicts residual correction:
        pred = baseline + residual
    """

    def __init__(self, in_channels=1536, hidden=64, out_channels=2, use_batch_norm=True):
        super().__init__()

        Norm = nn.BatchNorm3d if use_batch_norm else nn.InstanceNorm3d
        norm_kwargs = {"affine": True, "track_running_stats": True} if use_batch_norm else {"affine": True}

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, padding=0),
            Norm(hidden, **norm_kwargs),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            Norm(hidden, **norm_kwargs),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(hidden, 32, kernel_size=3, padding=1),
            Norm(32, **norm_kwargs),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(32, out_channels, kernel_size=1, padding=0),
        )

        self.init_last_layer_zero()

    def init_last_layer_zero(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x, baseline):
        residual = self.net(x)
        return baseline + residual


class GroupedRFResidualNet(nn.Module):
    """
    Receive-channel grouped fusion model.

    Input channel order:
        1536 = 128 receive channels × 3 angles × 4 components
    """

    def __init__(
        self,
        n_rx=128,
        n_angle=3,
        n_comp=4,
        per_rx_features=2,
        hidden=64,
        out_channels=2,
    ):
        super().__init__()

        in_channels = n_rx * n_angle * n_comp
        grouped_out = n_rx * per_rx_features

        self.rx_group_fusion = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=grouped_out,
                kernel_size=1,
                groups=n_rx,
                bias=True,
            ),
            nn.InstanceNorm3d(grouped_out, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.channel_fusion = nn.Sequential(
            nn.Conv3d(grouped_out, hidden, kernel_size=1, padding=0),
            nn.BatchNorm3d(hidden, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.spatial = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(hidden, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(32, out_channels, kernel_size=1, padding=0),
        )

        last = self.spatial[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x, baseline):
        feat = self.rx_group_fusion(x)
        feat = self.channel_fusion(feat)
        residual = self.spatial(feat)
        return baseline + residual


class ConvINAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=p),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvINAct(ch, ch, k=3, p=1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(ch, affine=True),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(x + out)


class EarlyMixWideResidualNet(nn.Module):
    """
    Wider early global channel-mixing model.
    """

    def __init__(self, in_channels=1536, stem_channels=128, mid_channels=64, out_channels=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, kernel_size=1, padding=0),
            nn.InstanceNorm3d(stem_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.res1 = ResidualBlock3D(stem_channels)
        self.res2 = ResidualBlock3D(stem_channels)

        self.down = nn.Sequential(
            nn.Conv3d(stem_channels, mid_channels, kernel_size=1, padding=0),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.res3 = ResidualBlock3D(mid_channels)
        self.out = nn.Conv3d(mid_channels, out_channels, kernel_size=1, padding=0)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, baseline):
        feat = self.stem(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.down(feat)
        feat = self.res3(feat)
        residual = self.out(feat)
        return baseline + residual


class WideDeepResidualBlockBN(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels, affine=True, track_running_stats=True),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class WideDeepResidualRFNet(nn.Module):
    """
    Full-resolution wider/deeper BatchNorm residual model.

    Input:
        x        [B,1536,Z,X,Y]
        baseline [B,2,Z,X,Y]

    Output:
        pred     [B,2,Z,X,Y]
    """

    def __init__(self, in_channels=1536, hidden=128, head_channels=32, out_channels=2, num_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, padding=0),
            nn.BatchNorm3d(hidden, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.blocks = nn.Sequential(*[WideDeepResidualBlockBN(hidden) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Conv3d(hidden, head_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(head_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(head_channels, out_channels, kernel_size=1, padding=0),
        )

        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x, baseline):
        feat = self.stem(x)
        feat = self.blocks(feat)
        residual = self.head(feat)
        return baseline + residual


class UNetResidualRFNet(nn.Module):
    """
    Anisotropic 3D U-Net residual model for 64x32x32 RF patches.

    Input:
        x        [B,1536,Z,X,Y]
        baseline [B,2,Z,X,Y]

    Output:
        pred     [B,2,Z,X,Y]

    The network preserves z resolution and downsamples only x/y using
    stride=(1,2,2). It predicts a residual correction:
        pred = baseline + residual
    """

    def __init__(self, in_channels=1536, base_ch=48, out_channels=2):
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=1, padding=0),
            nn.InstanceNorm3d(c1, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.enc1 = self._double_conv(c1, c1)
        self.down1 = self._down(c1, c2)

        self.enc2 = self._double_conv(c2, c2)
        self.down2 = self._down(c2, c3)

        self.enc3 = self._double_conv(c3, c3)
        self.down3 = self._down(c3, c3)

        self.bottleneck = self._double_conv(c3, c3)

        self.up3 = self._up(c3, c3)
        self.dec3 = self._double_conv(c3 + c3, c3)

        self.up2 = self._up(c3, c2)
        self.dec2 = self._double_conv(c2 + c2, c2)

        self.up1 = self._up(c2, c1)
        self.dec1 = self._double_conv(c1 + c1, c1)

        self.out = nn.Conv3d(c1, out_channels, kernel_size=1, padding=0)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _double_conv(in_ch, out_ch):
        return nn.Sequential(
            ConvINAct(in_ch, out_ch, k=3, p=1),
            ConvINAct(out_ch, out_ch, k=3, p=1),
        )

    @staticmethod
    def _down(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    @staticmethod
    def _up(in_ch, out_ch):
        return nn.ConvTranspose3d(
            in_ch,
            out_ch,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0,
        )

    @staticmethod
    def _concat_skip(x, skip, name):
        if x.shape[2:] != skip.shape[2:]:
            raise RuntimeError(
                f"{name} spatial mismatch: upsampled={tuple(x.shape[2:])}, "
                f"skip={tuple(skip.shape[2:])}"
            )
        return torch.cat([x, skip], dim=1)

    def forward(self, x, baseline):
        x = self.stem(x)

        s1 = self.enc1(x)
        x = self.down1(s1)

        s2 = self.enc2(x)
        x = self.down2(s2)

        s3 = self.enc3(x)
        x = self.down3(s3)

        x = self.bottleneck(x)

        x = self.up3(x)
        x = self.dec3(self._concat_skip(x, s3, "dec3"))

        x = self.up2(x)
        x = self.dec2(self._concat_skip(x, s2, "dec2"))

        x = self.up1(x)
        x = self.dec1(self._concat_skip(x, s1, "dec1"))

        residual = self.out(x)
        return baseline + residual


def _filter_init_kwargs(model_cls, kwargs):
    signature = inspect.signature(model_cls.__init__)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs

    accepted = {
        name
        for name, param in parameters.items()
        if name != "self"
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {name: value for name, value in kwargs.items() if name in accepted}


def build_model(model_name: str, **kwargs):
    name = model_name.lower()
    model_class = MODEL_NAME_TO_CLASS.get(name)

    if model_class == "TinyResidualRFNet":
        return TinyResidualRFNet(**_filter_init_kwargs(TinyResidualRFNet, kwargs))

    if model_class == "TinyBaselineConditionedRFNet":
        return TinyBaselineConditionedRFNet(**_filter_init_kwargs(TinyBaselineConditionedRFNet, kwargs))

    if model_class == "GroupedRFResidualNet":
        return GroupedRFResidualNet(**_filter_init_kwargs(GroupedRFResidualNet, kwargs))

    if model_class == "EarlyMixWideResidualNet":
        return EarlyMixWideResidualNet(**_filter_init_kwargs(EarlyMixWideResidualNet, kwargs))

    if model_class == "UNetResidualRFNet":
        return UNetResidualRFNet(**_filter_init_kwargs(UNetResidualRFNet, kwargs))

    if model_class == "WideDeepResidualRFNet":
        return WideDeepResidualRFNet(**_filter_init_kwargs(WideDeepResidualRFNet, kwargs))

    raise ValueError(f"Unknown model_name: {model_name}")


def _sanity_check_unet_residual_rf_net() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResidualRFNet().to(device).eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(2, 1536, 64, 32, 32, device=device)
    baseline = torch.randn(2, 2, 64, 32, 32, device=device)

    with torch.no_grad():
        y = model(x, baseline)

    max_abs_diff = torch.max(torch.abs(y - baseline)).item()

    print("UNetResidualRFNet sanity check")
    print(f"  device: {device}")
    print(f"  input shape    : {tuple(x.shape)}")
    print(f"  baseline shape : {tuple(baseline.shape)}")
    print(f"  output shape   : {tuple(y.shape)}")
    print("  downsample path: (64,32,32) -> (64,16,16) -> (64,8,8) -> (64,4,4)")
    print("  upsample path  : (64,4,4) -> (64,8,8) -> (64,16,16) -> (64,32,32)")
    print(f"  trainable parameters: {n_params:,} ({n_params / 1e6:.3f} M)")
    print(f"  max_abs(output - baseline): {max_abs_diff:.6e}")
    print(f"  output approximately baseline: {torch.allclose(y, baseline, atol=1e-6, rtol=1e-6)}")


if __name__ == "__main__":
    _sanity_check_unet_residual_rf_net()
