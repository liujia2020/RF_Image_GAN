# PROJECT_HANDOFF

## 1) 项目目标
用深度网络把 delay-aligned 多角度 RF tensor 映射到更接近目标 DAS 的复数 RF volume，并能在完整 carotid volume 上可靠 sliding-window 拼接和评估。

## 2) 数据链路
- 原始 RF / MATLAB 数据：例如 `Carotid_012.mat` 等 RF file。
- MATLAB 延时/声学处理：生成 delay-aligned RF tensor，包括 `F_RC_real/imag`、`F_CR_real/imag` 等多角度、多通道输入。
- Patch HDF5：`make_RF_learning_sample_v2` 写入 `input`、`label`、`baseline`、`meta/z_idx/x_idx/y_idx/source_file/frame_id`；MATLAB 索引是 1-based。
- Python Dataset：`RFLearningDataset` 读取 HDF5，transpose/stack 成 `[1536,Z,X,Y]`，输出 `label/baseline` 为 `[2,Z,X,Y]`，可按 sample scale normalize。
- Cache：`rf_cache_builder.py` 复用 `RFLearningDataset` 一次性解析，写入 `Data_cache*/{split}/input.dat/label.dat/baseline.dat/scale.dat/meta.npz`；训练时 `RFCachedDataset` 纯 memmap 读取，float16 存储、float32 返回。

## 3) 当前已完成实验和关键数字
| experiment | n | complex pred/base | complex impr | complex better | abs pred/base | abs impr | abs better | summary path |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `tiny_residual_full_2000` | 300 | 11291.51 / 7881.66 | 15.49% | 90.67% | 13943.46 / 8887.17 | 14.22% | 91.67% | `test_metrics/tiny_residual_full_2000/test_overall_summary.csv` |
| `tiny_nonpoint_full` | 225 | 3927.08 / 6334.01 | 44.36% | 95.11% | 3169.29 / 6540.32 | 51.63% | 99.11% | `test_metrics/tiny_nonpoint_full/test_overall_summary.csv` |
| `tiny_baseline_cond_nonpoint_full` | 225 | 3861.85 / 6334.01 | 45.43% | 95.11% | 3258.21 / 6540.32 | 49.65% | 99.11% | `test_metrics/tiny_baseline_cond_nonpoint_full/test_overall_summary.csv` |
| `tiny_baseline_cond_nonpoint_full_32` | 225 | 3665.31 / 6567.88 | 47.07% | 99.56% | 3507.49 / 6749.45 | 48.98% | 98.22% | `test_metrics/tiny_baseline_cond_nonpoint_full_32/test_overall_summary.csv` |
| `tiny_nonpoint_full_32` | 225 | 3668.72 / 6567.88 | 47.11% | 99.56% | 3458.71 / 6749.45 | 49.62% | 97.78% | `test_metrics/tiny_nonpoint_full_32/test_overall_summary.csv` |
| `tiny_random64_pilot80` | 12 | 2502.72 / 5163.75 | 51.39% | 100.00% | 2200.87 / 5832.64 | 62.73% | 100.00% | `test_metrics/tiny_random64_pilot80/test_overall_summary.csv` |
| `tiny_random64_full1500` | 225 | 3070.44 / 5614.82 | 47.32% | 97.33% | 2782.30 / 5886.37 | 52.50% | 98.22% | `test_metrics/tiny_random64_full1500/test_overall_summary.csv` |
| `tiny_bn_random64_full1500` | 225 | 2401.7 / 5614.8 | 56.85% | 100.00% | 2760.2 / 5886.4 | 51.85% | 100.00% | `test_metrics/tiny_bn_random64_full1500/test_overall_summary.csv` |

Per-category 摘要：
- `tiny_nonpoint_full_32`: carotid complex 42.31%, muscle 48.60%, phantom 50.41%；abs 分别 43.50%、56.07%、49.29%。
- `tiny_random64_pilot80`: carotid complex 53.61%, muscle 51.01%, phantom 49.54%；abs 分别 66.33%、59.48%、62.39%。
- `tiny_random64_full1500`: carotid complex 48.34%, muscle 45.71%, phantom 47.90%；abs 分别 56.32%、50.18%、51.00%。
- `tiny_bn_random64_full1500`: carotid complex ~53.5%, muscle ~53.2%, phantom ~48.9%；abs 分别 ~53.5%、~53.2%、~48.9%。
- BN 13 卷 dense64 诊断（`test_metrics/tiny_bn_random64_full1500/bn_multivolume_diagnostics_13.csv`）：
  carotid/muscle 非过平滑且全面优于 IN；phantom 有一致轻微细节损失但可接受，BN 方案定稿采用。

## 4) 关键模型 checkpoint 索引
| experiment | model_class | best_epoch | best_val_l1 | checkpoint |
|---|---|---:|---:|---|
| `tiny_baseline_cond_nonpoint_full` | `TinyBaselineConditionedRFNet` | 90 | 0.148312 | `checkpoint/tiny_baseline_cond_nonpoint_full/best_model.pth` |
| `tiny_baseline_cond_nonpoint_full_32` | `TinyBaselineConditionedRFNet` | 27 | 0.144972 | `checkpoint/tiny_baseline_cond_nonpoint_full_32/best_model.pth` |
| `tiny_nonpoint_full` | `TinyResidualRFNet` | 84 | 0.148359 | `checkpoint/tiny_nonpoint_full/best_model.pth` |
| `tiny_nonpoint_full_32` | `TinyResidualRFNet` | 4 | 0.147047 | `checkpoint/tiny_nonpoint_full_32/best_model.pth` |
| `tiny_random64_pilot80` | `TinyResidualRFNet` | 10 | 0.139059 | `checkpoint/tiny_random64_pilot80/best_model.pth` |
| `tiny_random64_full1500` | `TinyResidualRFNet` | 41 | 0.147535 | `checkpoint/tiny_random64_full1500/best_model.pth` |
| `tiny_bn_random64_full1500` | `TinyResidualRFNet` | 14 | 0.125628 | `checkpoint/tiny_bn_random64_full1500/best_model.pth` |

## 5) 关键拼图产物索引
| experiment | volume dir | source/frame | mode | patch/stride | z coverage | shape | L1 pred/base |
|---|---|---|---|---|---|---|---:|
| `tiny_nonpoint_full_32` | `vis_best_model/tiny_nonpoint_full_32/full_volume` | `Carotid_012.mat` / 1 | dense32_nooverlap_root | 32x16x16 / no overlap | full | `(2, 1024, 128, 128)` | 旧 dense32 对照 |
| `tiny_nonpoint_full_32` | `vis_best_model/tiny_nonpoint_full_32/full_volume/Carotid_012_frame1_overlap25_zband150-600` | `Carotid_012.mat` / 1 | overlap25_hann_zband | `(32,16,16)` / `(24,12,12)` | 1-based 150-600 | `(2, 451, 128, 128)` | z-band 对照 |
| `tiny_random64_pilot80` | `vis_best_model/tiny_random64_pilot80/full_volume/Carotid_012_frame1_dense64_nooverlap` | `Carotid_012.mat` / 1 | dense64_nooverlap | `(64,32,32)` / `(64,32,32)` | full | `(2, 1024, 128, 128)` | pred 7554.03 / base 12872.53 |
| `tiny_random64_full1500` | `vis_best_model/tiny_random64_full1500/full_volume/Carotid_012_frame1_dense64_nooverlap` | `Carotid_012.mat` / 1 | dense64_nooverlap | `(64,32,32)` / `(64,32,32)` | full | `(2, 1024, 128, 128)` | pred 6542.23 / base 12872.53 |
| `tiny_bn_random64_full1500` | `vis_best_model/tiny_bn_random64_full1500/full_volume` | `Carotid_012.mat` / 1 | dense64_nooverlap | `(64,32,32)` / `(64,32,32)` | full | `(2, 1024, 128, 128)` | pred 5232 / base 12872.53 |

## 6) 已确认的工程教训
此前 Jupyter 单 cell 重跑导致 kernel 里残留旧变量，EXP 写着 Tiny，实际却训练 baseline-conditioned，并把 checkpoint 写进旧目录；现在训练入口会在 startup 打印 experiment_name、ckpt_dir、真实 `type(model).__name__` 和参数量，并 assert 实验名与 checkpoint 路径、model_name 与模型类一致。HDF5 每个 epoch 重读和 transpose/stack 是主要 I/O 瓶颈，64x32x32 full1500 采用 float16 memmap cache 后才可稳定训练。Sliding-window 拼接必须先做 coverage check，确认每个 voxel 覆盖次数正确，再谈模型效果。MATLAB meta 是 1-based，Python 切片前必须减 1。图像观感可能被 dB ref、db_min、aspect 影响；当视觉和结论冲突时，以 full-volume L1、边界跳变等客观指标仲裁。

## 7) 主要文件清单
- `rf_learning_dataset.py`: 原始 patch HDF5 Dataset，动态读取 input/label/baseline/scale/meta。
- `rf_cache_builder.py`: HDF5 到 memmap cache 构建器，含 20-sample correctness self-test。
- `rf_cached_dataset.py`: memmap cache Dataset，drop-in 返回 input/label/baseline/scale/path/category/idx。
- `rf_models.py`: Tiny、baseline-conditioned、grouped、earlymix、UNet 模型和 `build_model` registry。
- `rf_train_utils.py`: 训练循环、startup 自检、checkpoint/config 保存、early stopping。
- `rf_eval_utils.py`: test set 评估、denormalize、overall/per-category/worse samples 汇总。
- `rf_visualization.py`: patch 级 B-mode 可视化、slice/dB/aspect 工具。
- `rf_stitch.py`: dense sliding-window coverage check、streaming inference、volume stitching。
- `rf_stitch_vis.py`: full-volume B-mode 对比图脚本。
- `rf_bn_multivolume_diagnostics.py`: IN vs BN 多卷 seam/过平滑统一诊断脚本，输出视觉图、seam 指标和非边界高频保真指标。
- `build_RF_datase_V2.m`: MATLAB patch HDF5 生成主脚本/入口之一，含 dense/random 采样配置。
- `build_RF_random64_pilot80.m`: 64x32x32 pilot 随机 patch 数据生成脚本。
- `build_RF_random64_full1500.m`: 64x32x32 full1500 数据生成脚本，file-level split。
- `8tiny_nonpoint_full_32.ipynb`: 32x16x16 普通 Tiny cache 训练/评估 notebook。
- `9tiny_random64_pilot80.ipynb`: 64x32x32 pilot Tiny notebook。
- `10tiny_random64_full1500.ipynb`: 64x32x32 full1500 Tiny cache 训练/评估 notebook。
- `11unet_random64_full1500.ipynb`: 64x32x32 full1500 UNet 候选 notebook，已完成对照且未赢 Tiny。

## 动态范围线程(回归范式)——人眼验证否决,转 conditional GAN [2026-06 修正]

判决:
- 不是"已收官/交付配方"。回归配方(WideDeep+SSIM+crop8)只改善了代理指标,nii + 3D Slicer 人眼验证下失败:phantom 线靶丢失(label 有 pred 无,已确认);组织 speckle 形态差(均值回归 -> mush)。
- 范式级根因:回归逼近条件均值,产不出随机散斑;稀疏强反射体(点靶)被对冲压平。
- 留得下(范式无关):缝根因#1 的 BN 修正、数据管线、拼接/crop8/halo、诊断纪律。
- 作废:"SSIM+wide+crop8=交付配方/组织上成立"。重归类为负结果:必须换生成式的证据。
- 决策:转 conditional GAN(对抗=真实散斑 + 保真=锁相干结构/防幻觉),主攻组织+仿体真实散斑;点靶简单留后,simu_point 暂不进训练;扩散因 3D+8GB 吃不消列 future work。
- 强制:任何"质量达标"结论必须过验证关卡(见 claude.md)。

以下为回归范式的代理指标(slope/p99/abs_std/SSIM),已被人眼 3D 验证否决,见判决。

### 配方四要素
- 模型:WideDeepResidualRFNet(全程 BatchNorm3d,~3.74M)。stem Conv3d(1536->128,1x1)+BN+LReLU;4x 全分辨率残差块[Conv3d(128,128,3)+BN+LReLU -> Conv3d(128,128,3)+BN,skip,LReLU];head Conv3d(128->32,1x1)+BN+LReLU -> Conv3d(32->2,1x1),末层 zero-init;pred=baseline+residual。
- 损失:complex_L1 + 0.1*abs(包络)_L1 + 1.0*(1-SSIM(包络));SSIM=自实现 3D 高斯窗 win=7、nonnegative、data_range 按样本 label 包络 max(归一化空间 ~1.0-1.1)。
- 选模:best_by_val_ssim(不是 val_l1)。
- 拼接:crop8 center(每 patch 各向丢 8 voxel 污染边)+ 足够 overlap;等价地可用 halo 推理(从源体读 patch+8 halo、只留中心),无需存整套 overlap patch;z-band 流式控制存储。

### 验证数(wide+SSIM-crop8 vs BN-L1;指标为 pred/label 比)
以下表格仅作为技术记录保留,不得再解释为交付达标。
| 卷 | x seam | xz_y seam | p99 | abs_std | deep p99 | deep abs_std | nonbd cplx impr | nonbd abs impr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Carotid_008 | 0.83 | 0.77 | 0.82 | 0.84 | 0.95 | 0.95 | 0.49(BN0.60) | 0.65(BN0.58) |
| Carotid_073 | 0.77 | 0.67 | 0.76 | 0.79 | 0.92 | 0.92 | 0.48 | 0.64 |
| Muscle_056 | 0.78 | 0.81 | 0.97 | 0.94 | 0.96 | 0.95 | 0.49 | 0.66 |
| Phantom_036 | 0.84 | 0.89 | 0.89 | 0.91 | 1.19(过冲) | 1.10(过冲) | 0.46 | 0.62 |

对照 BN-L1:p99 ~0.24-0.36、deep ~0.31(压垮)。所有 wide+SSIM:x=32/64/96 无 >1 周期峰(缝已协议修复)。

### 开放项
1. phantom 深部低信号过冲(deep p99 1.19、abs_std 1.10):realism 目标把组织典型深部幅度外推到 OOD 的 phantom 低信号区 -> 多加了对比度。轻、局部、phantom 为标定体。判为低优先,暂不为它重开损失调参(治它可能回吐组织收益)。如需治:category-aware 或惩罚过冲。
2. realism vs complex-L1 trade:非边界 complex_impr 由 ~0.57 降到 ~0.47,是主动用 L1 换真实度,接受。
3. 推广验证用的是 halo 拼装的"虚拟 crop8"(非真 overlap HDF5),与 Carotid_008 真 overlap 结果一致、互证;若要 bit 级严谨,可对 phantom 补一次真 overlap。
4. 广度:已验证 4 卷(carotidx2 / musclex1 / phantomx1)、各 1 帧;更多卷/帧/组织可进一步增强稳健性。

## 8) 已知但未走完的分支
- `11unet_random64_full1500.ipynb` 已完成训练/评估;UNet patch test 与拼图边界跳变均未赢 Tiny。
- 缝根因 #1（InstanceNorm per-patch 统计不连续）已由 BN 重训根除：
  `tiny_bn_random64_full1500`（InstanceNorm3d→BatchNorm3d），seam 1.46→0.535x（-63%），
  patch/full-volume 精度同步升 ~20%，边界/非边界 L1 等比例降，排除过度平滑。
  注意：此前文档建议"InstanceNorm 改 track_running_stats"是错误方案（仍为 instance-level），
  正解是换成 BatchNorm3d。详见 claude.md。
- 数据 b 空间不连续仍存在（baseline seam 约 2.8-3.0x），但 13 卷 BN 诊断后确认它在 pred
  层面未成为主瓶颈，已降为低优先；overlap/大块仅作为未来候选。
- **多卷 seam 泛化**（2026-05-30）：MATLAB batch 生成 4 卷新 dense64（carotid×2 + muscle×1 + phantom×1），
  BN 在全部 5 卷上 seam 均低于 IN。carotid/muscle 上治缝确凿（IN seam>1, BN 压降且 L1 同步改善）。
  phantom 上 IN seam<1（无伪缝），BN 仍压降但 L1 改善骤减（-6.6% vs carotid -23%），疑为过度平滑。
  详见 EXPERIMENTS_LOG.md。
- **2026-05-31 #2 前置判断**：Carotid_012/carotid 上，BN 后 pred 层面基本无视觉缝，
  full-volume x-boundary jump 为 0.50x label（xz y=64 为 0.61x），而 baseline 仍为 2.87x label。
  这说明 baseline 空间不连续未显著传染到 BN pred，#2 在 carotid 上降为低优先级。
  但 phantom/muscle 不能外推；#2 残余空间不连续和 phantom/abs 过平滑疑点是同一问题两面，
  等新 dense 到位后用 `rf_bn_multivolume_diagnostics.py` 统一判读。
- **2026-06-01 13 卷 BN 诊断定稿**：dense64 共 carotid5/muscle4/phantom4，raw patch 经
  `dense64_index.csv` 核对后全可读（F 盘 WSL 9p 断挂已重挂恢复）。BN seam 三类全面压低：
  carotid 0.496、muscle 0.508、phantom 0.458。carotid/muscle 非边界 complex improvement
  分别为 0.586/0.592，非过平滑；phantom 为 0.565，abs 0.520、HF RMS/label 0.436，
  相比 carotid 0.586/0.554/0.472 低一档且 4 卷一致，定性为系统性轻微细节损失、
  可接受 trade-off，非 deal-breaker。BN 方案定稿采用，#1 收官；当前不为 phantom 另开工程。
- 25% overlap + Hann 只在 32x16x16 z-band 上验证过，尚未扩展到 64x32x32 或完整 volume。
- dense full-volume 已扩展到 13 卷（carotid5/muscle4/phantom4）用于 BN seam/过平滑诊断。
- full1500 Tiny train 抽样 improvement 约 56.4%，test 约 47.3%，存在 gap 但未见断崖式过拟合。
