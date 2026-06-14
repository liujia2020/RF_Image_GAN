import numpy as np
import torch
torch.set_grad_enabled(False)
import sys
sys.path.insert(0, '/media/liujia/8CC24D13C24D02C6/code/RF_Image_GAN')
from rf_cgan_losses import complex_envelope, STRUCT_NORMALIZATION_EPS

cache_dir = '/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607/test'

meta = np.load(f'{cache_dir}/meta.npz', allow_pickle=True)
categories = meta['category']
N = len(categories)
per_sample_label = 2 * 64 * 32 * 32  # 131072

label_mm = np.memmap(f'{cache_dir}/label.dat', dtype=np.float16, mode='r')
baseline_mm = np.memmap(f'{cache_dir}/baseline.dat', dtype=np.float16, mode='r')
scale_mm = np.memmap(f'{cache_dir}/scale.dat', dtype=np.float32, mode='r')

def load_label(idx):
    off = idx * per_sample_label
    raw = np.array(label_mm[off:off+per_sample_label], dtype=np.float32)
    return torch.from_numpy(raw).reshape(2, 64, 32, 32)

def load_baseline(idx):
    off = idx * per_sample_label
    raw = np.array(baseline_mm[off:off+per_sample_label], dtype=np.float32)
    return torch.from_numpy(raw).reshape(2, 64, 32, 32)

indices = {}
for i in range(N):
    cat = str(categories[i])
    if cat not in indices:
        indices[cat] = i
print(f'Selected indices: {indices}')

print()
print(f'{"cat":10s} {"idx":5s} {"s":12s} {"carrier_old":16s} {"carrier_old/s":18s} {"carrier_new":18s} {"abs_diff":10s} {"rel_diff":10s}')
print('-' * 120)

for cat, idx in sorted(indices.items()):
    s_val = float(scale_mm[idx])
    label = load_label(idx) * s_val
    label = label.unsqueeze(0)
    pred = load_baseline(idx) * s_val
    pred = pred.unsqueeze(0)

    # carrier_old: direct complex L1
    carrier_old = torch.nn.functional.l1_loss(pred, label)

    # s = mean(|env(label)|) + eps
    label_env = complex_envelope(label)
    s = label_env.detach().abs().mean() + STRUCT_NORMALIZATION_EPS

    # carrier_new: normalized
    carrier_new = torch.nn.functional.l1_loss(pred / s, label / s)

    carrier_old_div_s = carrier_old / s

    abs_diff = float((carrier_new - carrier_old_div_s).abs())
    rel_diff = float(abs_diff / (float(carrier_old_div_s.abs()) + 1e-12))

    print(f'{cat:10s} {idx:5d} {float(s):12.4f} {float(carrier_old):16.6f} {float(carrier_old_div_s):18.12f} {float(carrier_new):18.12f} {abs_diff:10.2e} {rel_diff:10.2e}')

print()
print('=== Carrier unnormalized reconstruction ===')
for cat, idx in sorted(indices.items()):
    s_val = float(scale_mm[idx])
    label = load_label(idx) * s_val
    label = label.unsqueeze(0)
    pred = load_baseline(idx) * s_val
    pred = pred.unsqueeze(0)
    label_env = complex_envelope(label)
    s = label_env.detach().abs().mean() + STRUCT_NORMALIZATION_EPS
    carrier_old = torch.nn.functional.l1_loss(pred, label)
    carrier_new = torch.nn.functional.l1_loss(pred / s, label / s)
    reconstructed = carrier_new * s
    diff = float(abs(carrier_old - reconstructed))
    print(f'{cat:10s}: carrier_old={float(carrier_old):.8f}, carrier_new*s={float(reconstructed):.8f}, diff={diff:.2e}')

print()
print('=== Struct normalization still intact ===')
from rf_cgan_losses import AnisotropicGaussianLowpass3D
lowpass = AnisotropicGaussianLowpass3D(sigma_zyx=(6.0, 3.0, 3.0), truncate=3.0)
for cat, idx in sorted(indices.items()):
    s_val = float(scale_mm[idx])
    label = load_label(idx) * s_val
    label = label.unsqueeze(0)
    pred = load_baseline(idx) * s_val
    pred = pred.unsqueeze(0)
    pred_env = complex_envelope(pred)
    label_env = complex_envelope(label)
    s = label_env.detach().abs().mean() + STRUCT_NORMALIZATION_EPS
    struct_old = torch.mean(torch.abs(lowpass(pred_env - label_env)))
    struct_new = torch.mean(torch.abs(lowpass((pred_env - label_env) / s)))
    abs_diff = float((struct_new - struct_old / s).abs())
    rel_diff = float(abs_diff / (float(struct_old / s).abs() + 1e-12))
    print(f'{cat:10s}: struct_old={float(struct_old):.6f}, struct_new={float(struct_new):.6f}, struct_old/s={float(struct_old/s):.8f}, rel_diff={rel_diff:.2e}')

print()
print('=== Magnitude check: all three losses O(1)? ===')
for cat, idx in sorted(indices.items()):
    s_val = float(scale_mm[idx])
    label = load_label(idx) * s_val
    label = label.unsqueeze(0)
    pred = load_baseline(idx) * s_val
    pred = pred.unsqueeze(0)
    label_env = complex_envelope(label)
    s = label_env.detach().abs().mean() + STRUCT_NORMALIZATION_EPS

    # adv ~ O(1) by design (LSGAN probability loss)
    # struct normalized
    struct_val = torch.mean(torch.abs(lowpass(complex_envelope(pred / s) - label_env / s)))
    # Actually let me use the same computation: pred_env/s vs label_env/s
    pred_env = complex_envelope(pred)
    struct_new = torch.mean(torch.abs(lowpass((pred_env - label_env) / s)))
    # carrier normalized
    carrier_new = torch.nn.functional.l1_loss(pred / s, label / s)

    print(f'{cat:10s}: struct_w={float(struct_new):.4f} (O(1)), carrier_w={float(carrier_new * 0.25):.4f} (x0.25), NaN={bool(torch.isnan(struct_new) or torch.isnan(carrier_new))}')
