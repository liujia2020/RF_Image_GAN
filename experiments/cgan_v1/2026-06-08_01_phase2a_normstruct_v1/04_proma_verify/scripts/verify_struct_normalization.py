import numpy as np
import torch
torch.set_grad_enabled(False)
import sys
sys.path.insert(0, '/media/liujia/8CC24D13C24D02C6/code/RF_Image_GAN')
from rf_cgan_losses import complex_envelope, AnisotropicGaussianLowpass3D, STRUCT_NORMALIZATION_EPS

cache_dir = '/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607/test'

meta = np.load(f'{cache_dir}/meta.npz', allow_pickle=True)
categories = meta['category']
N = len(categories)
label_shape = tuple(meta['label_shape'])  # [N, 2, 64, 32, 32]
baseline_shape = tuple(meta['baseline_shape'])
print(f'N={N}, label_shape={label_shape}')

per_sample_label = 2 * 64 * 32 * 32  # 131072
per_sample_input = 1536 * 64 * 32 * 32

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

lowpass = AnisotropicGaussianLowpass3D(sigma_zyx=(6.0, 3.0, 3.0), truncate=3.0)

print()
print(f'{"cat":10s} {"idx":5s} {"s":12s} {"struct_old":14s} {"struct_old/s":18s} {"struct_new":18s} {"abs_diff":10s} {"rel_diff":10s}')
print('-' * 115)

for cat, idx in sorted(indices.items()):
    s_val = float(scale_mm[idx])
    label = load_label(idx) * s_val
    label = label.unsqueeze(0)
    pred = load_baseline(idx) * s_val
    pred = pred.unsqueeze(0)
    
    pred_env = complex_envelope(pred)
    label_env = complex_envelope(label)
    
    struct_old = torch.mean(torch.abs(lowpass(pred_env - label_env)))
    s = label_env.detach().abs().mean() + STRUCT_NORMALIZATION_EPS
    struct_new = torch.mean(torch.abs(lowpass((pred_env - label_env) / s)))
    struct_old_div_s = struct_old / s
    
    abs_diff = float((struct_new - struct_old_div_s).abs())
    rel_diff = float(abs_diff / (float(struct_old_div_s.abs()) + 1e-12))
    
    print(f'{cat:10s} {idx:5d} {float(s):12.4f} {float(struct_old):14.6f} {float(struct_old_div_s):18.14f} {float(struct_new):18.14f} {abs_diff:10.2e} {rel_diff:10.2e}')

print()
print('=== struct_unnormalized reconstruction ===')
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
    reconstructed = struct_new * s
    diff = float(abs(struct_old - reconstructed))
    print(f'{cat:10s}: struct_old={float(struct_old):.8f}, struct_new*s={float(reconstructed):.8f}, diff={diff:.2e}')
