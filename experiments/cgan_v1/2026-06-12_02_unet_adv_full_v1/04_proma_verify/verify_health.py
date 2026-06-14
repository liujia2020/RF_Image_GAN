"""Proma 独立验证：健康度复算 + phantom speckle + 伪影检查。不碰生产代码路径。"""
import json, csv, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

RUN = Path('/media/liujia/8CC24D13C24D02C6/code/RF_Image_GAN/experiments/cgan_v1/2026-06-12_02_unet_adv_full_v1')
TRAIN = RUN / '02_train'
METRICS = TRAIN / 'metrics'
FIG = TRAIN / 'figures'
CKPT = TRAIN / 'checkpoints'
OUT = RUN / '04_proma_verify'
OUT.mkdir(parents=True, exist_ok=True)

# ---- 固定 val 样本（同 notebook cell1 输出） ----
FIXED_VAL_INDICES = {
    'carotid': [0, 1, 2, 3, 4, 5, 6, 7],
    'muscle':  [350, 351, 352, 353, 354, 355, 356, 357],
    'phantom': [700, 701, 702, 703, 704, 705, 706, 707],
}
VAL_INDICES = sorted(sum(FIXED_VAL_INDICES.values(), []))

# ---- 数据路径 ----
BMODE_DIR = Path('/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_bmode_gt_ref64407p58_20260610')
VAL_DIR = BMODE_DIR / 'val'

# ---- 加载 checkpoint epoch 25（有 D 的最后一次；epoch 42 没存 ckpt） ----
CKPT_PATH = CKPT / 'epoch025.pt'
print(f'[1/6] Loading checkpoint: {CKPT_PATH}')
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
print(f'  Keys: {list(ckpt.keys())}')
print(f'  G keys: {list(ckpt["generator_state_dict"].keys())[:3]}...')
print(f'  D keys: {list(ckpt["discriminator_state_dict"].keys())[:3]}...')

# ---- 手动构建 D（不 import 项目代码，直接复现） ----
class BMode3DPatchDiscriminator(torch.nn.Module):
    """与 rf_cgan_models.py 一致：3D 条件 PatchGAN，输入 [B,2,64,32,32]。"""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(2, 64, 4, 2, 1), torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv3d(64, 128, 4, 2, 1), torch.nn.InstanceNorm3d(128, affine=True),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv3d(128, 256, 4, 2, 1), torch.nn.InstanceNorm3d(256, affine=True),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv3d(256, 1, 3, 1, 1),
        )
    def forward(self, x):
        return self.net(x)

d = BMode3DPatchDiscriminator()
d.load_state_dict(ckpt['discriminator_state_dict'])
d.eval()

# 参数计数
d_params = sum(p.numel() for p in d.parameters())
print(f'  D params: {d_params:,} (expected ~1,490,000)')

print(f'\n[2/6] Loading fixed val data ({len(VAL_INDICES)} samples)...')
gts, baselines = [], []
for idx in VAL_INDICES:
    gt = np.array(np.memmap(VAL_DIR / 'label_bmode.dat', dtype=np.float16, mode='r',
                            shape=(500, 1, 64, 32, 32))[idx], dtype=np.float32, copy=True)
    bl = np.array(np.memmap(VAL_DIR.parent.parent / 'fp16_cache' / 'val' / 'baseline.dat',
                            dtype=np.float16, mode='r',
                            shape=(500, 1, 64, 32, 32))[idx], dtype=np.float32, copy=True)
    # baseline 是 raw RF 复数 [2,...]，需转 B-mode
    # 但这里 baseline 已经是 B-mode 格式 — 检查形状
    gts.append(gt)
    baselines.append(bl)

print(f'  gt shape: {gts[0].shape}, baseline shape: {baselines[0].shape}')

# ---- 加载 G ----
class DoubleConv3D(torch.nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, stride, 1, bias=False),
            torch.nn.BatchNorm3d(out_ch, momentum=0.9),
            torch.nn.ReLU(True),
            torch.nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False),
            torch.nn.BatchNorm3d(out_ch, momentum=0.9),
            torch.nn.ReLU(True),
        )
    def forward(self, x):
        return self.conv(x)

class Light3DUNet(torch.nn.Module):
    def __init__(self, in_channels=1536, out_channels=1):
        super().__init__()
        self.head = torch.nn.Conv3d(in_channels, 64, 1, bias=False)
        self.enc1 = DoubleConv3D(64, 64)
        self.down1 = torch.nn.Conv3d(64, 48, (1, 2, 3), (1, 1, 2), bias=False)
        self.enc2 = DoubleConv3D(48, 48)
        self.down2 = torch.nn.Conv3d(48, 56, (2, 2, 2), (1, 1, 1), bias=False)
        self.bridge = DoubleConv3D(56, 64)
        self.up2 = torch.nn.ConvTranspose3d(64, 56, (2, 2, 2), (1, 1, 1), bias=False)
        self.dec2 = DoubleConv3D(104, 48)
        self.up1 = torch.nn.ConvTranspose3d(48, 48, (1, 2, 3), (1, 1, 2), bias=False)
        self.dec1 = DoubleConv3D(96, 64)
        self.tail = torch.nn.Conv3d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(self.head(x))
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        b = self.bridge(d2)
        u2 = self.up2(b)
        d2_out = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2_out)
        d1_out = self.dec1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.tail(d1_out))

g = Light3DUNet()
g.load_state_dict(ckpt['generator_state_dict'])
g.eval()
g_params = sum(p.numel() for p in g.parameters())
print(f'  G params: {g_params:,} (expected ~2,110,000)')
