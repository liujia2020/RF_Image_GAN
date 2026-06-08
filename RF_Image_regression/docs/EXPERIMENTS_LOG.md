# EXPERIMENTS_LOG

按 `checkpoint/*/training_history.csv` 修改时间排序；训练耗时为文件时间跨度估计，若 notebook/日志未记录完整 wall-clock，则标注为估计或未记录。

## tiny_residual_full_2000
- 数据配方：早期 32x16x16 full_2000 split，含 `simu_point`。
- 模型：TinyResidualRFNet。
- 训练：last_epoch=100；checkpoint 缺少当前标准 `best_model.pth` 字典索引。
- Test overall：n=300，complex 11291.51 / 7881.66，improvement 15.49%，better 90.67%；abs improvement 14.22%，abs better 91.67%。
- 备注：`simu_point` 类严重拉低整体，非当前 nonpoint 主线。

## tiny_nonpoint_full
- 数据配方：32x16x16 nonpoint full split。
- 模型：TinyResidualRFNet。
- 训练：last_epoch=100，best_epoch=84，best_val_l1=0.148359。
- Test overall：n=225，complex 3927.08 / 6334.01，improvement 44.36%，better 95.11%；abs improvement 51.63%，abs better 99.11%。
- 备注：普通 Tiny 旧版 nonpoint 对照。

## tiny_baseline_cond_nonpoint_full
- 数据配方：32x16x16 nonpoint full split。
- 模型：TinyBaselineConditionedRFNet。
- 训练：last_epoch=100，best_epoch=90，best_val_l1=0.148312。
- Test overall：n=225，complex 3861.85 / 6334.01，improvement 45.43%，better 95.11%；abs improvement 49.65%，abs better 99.11%。
- 备注：baseline-conditioned 消融，complex 略高于 `tiny_nonpoint_full`。

## tiny_baseline_cond_nonpoint_full_32
- 数据配方：32x16x16 nonpoint full split。
- 模型：TinyBaselineConditionedRFNet。
- 训练：last_epoch=100，best_epoch=27，best_val_l1=0.144972。
- Test overall：n=225，complex 3665.31 / 6567.88，improvement 47.07%，better 99.56%；abs improvement 48.98%，abs better 98.22%。
- 备注：32x16x16 baseline-conditioned 对照。

## tiny_nonpoint_full_32
- 数据配方：32x16x16 nonpoint full split，cache 训练。
- 模型：TinyResidualRFNet。
- 训练：last_epoch=19，best_epoch=4，best_val_l1=0.147047。
- Test overall：n=225，complex 3668.72 / 6567.88，improvement 47.11%，better 99.56%；abs improvement 49.62%，abs better 97.78%。
- 备注：32x16x16 普通 Tiny 基准线，dense32 full-volume 拼图来源。

## tiny_random64_pilot80
- 数据配方：`RF_LearningSamples_random64_pilot80`；64x32x32 pilot，train/val/test=57/12/12，未用 cache。
- 模型：TinyResidualRFNet。
- 训练：last_epoch=25，best_epoch=10，best_val_l1=0.139059。
- Test overall：n=12，complex 2502.72 / 5163.75，improvement 51.39%，better 100.00%；abs improvement 62.73%，abs better 100.00%。
- 备注：用于验证 64x32x32 大 patch 方向和 dense64 拼图。

## tiny_random64_full1500
- 数据配方：`RF_LearningSamples_random64_full1500` 到 `Data_cache_random64_full1500`；64x32x32，300 files x 5 patches，file-level 70/15/15。
- 模型：TinyResidualRFNet。
- 训练：原训练因电脑重启中断于 epoch 35；从 best checkpoint epoch 31 恢复，early stopping 于 epoch 56；best_epoch=41，best_val_l1=0.147535。
- Test overall：n=225，complex 3070.44 / 5614.82，improvement 47.32%，better 97.33%；abs improvement 52.50%，abs better 98.22%。
- 备注：当前强 Tiny baseline；dense64 full-volume L1 为 pred 6542.23、baseline 12872.53；x-boundary pred/label jump ratio 约 1.46x。

## tiny_bn_random64_full1500
- 数据配方：同 `tiny_random64_full1500`，使用 `Data_cache_random64_full1500`。
- 改动：`TinyResidualRFNet` 的 3 个 `InstanceNorm3d(affine=True)` → `BatchNorm3d(affine=True, track_running_stats=True)`，唯一主变量；旁变量 `batch_size` 4→8（RTX 5060 8GB 显存约束）。
- 训练：`num_workers=0`（WSL 大张量 IPC 不兼容，诊断后回归），early stopping 于 epoch 25；best_epoch=14，best_val_l1=0.125628。
- Test overall：n=225，complex 2401.7 / 5614.8，improvement 56.85%，better 100.00%；abs 2760.2 / 5886.4，improvement 51.85%，abs better 100.00%。
- Per-category complex improvement：carotid ~53.5%、muscle ~53.2%、phantom ~48.9%。
- Full-volume dense64 no-overlap (Carotid_012 frame1)：pred L1 5232、baseline 12872（-20% vs old Tiny 6542）；seam pred/label 0.535x（old Tiny 1.46x，UNet 1.53x）。
- 验证（排除过度平滑）：边界/非边界 L1 等比例降 ~20%；穿边界波形显示 IN 在个别位置引入 45x/9.7x 假台阶被 BN 压回 ~0.85x，跟随 label 真实结构；个别位置（z=512 处 step/label=0.33x）有轻微过平滑，无损总账。
- 判决：InstanceNorm3d → BatchNorm3d 根除缝根因 #1（IN per-patch 统计不连续），精度同步提升，非以过平滑换 seam。seam 0.535x < 1 的正确解读：IN 假台阶消除使均值大幅下降，非"BN 比 label 更好"。
- 备注：当前强 baseline，全面超越 `tiny_random64_full1500`（IN）。后续 13 卷诊断确认 BN 方案定稿采用，
  数据 b 空间不连续在 BN pred 层面降为低优先。

### 泛化验证（2026-05-30，patch 级，N=45 文件）

- 用 225 个 test patch 按 RF 源文件分组（45 文件×5 patches），对比 IN vs BN 的 per-file improvement。
- **complex improvement**：45 文件 BN 全面优于 IN，零回归。
  carotid +9.7pp、muscle +11.4pp、phantom +7.5pp（per-file 均值）。
  精度泛化成立。
- **abs improvement**：26/45 文件 BN 低于 IN，最差 RF000490_carotid -10.1pp。
  BN 在 complex 全面提升，但 abs 部分文件系统性回归（最差 -10pp），
  疑为 BN 优化复数主成分时损失部分幅度信息，原因待查。
- **边界**：此为 patch 级精度泛化，非 seam 治缝泛化。
  patch 指标对 block 缝不敏感（claude.md 纪律 #3）。
  seam 泛化需多 volume dense 数据，目前仅 Carotid_012 frame1（N=1）。
- **待决策（留 codex）**：生成多 volume dense64 需 MATLAB ~1h/卷，
  4 卷 4-5h 人工操作（Windows MATLAB GUI，需切换源 volume 参数），
  是否投入做 seam 泛化验证由 codex 根据优先级判。

### 多卷 seam 泛化验证（2026-05-30，5 卷 dense64）

- 生成 4 卷新 dense64（MATLAB batch 5.3h）：carotid RF000487/RF000488、muscle RF000386、phantom RF000587，加上已有的 Carotid_012 共 5 卷。
- 每卷跑 IN vs BN 的 dense64 no-overlap 拼图 + seam_metric（x=32/64/96）。
- **结果：BN 在全部 5 卷上 seam 均低于 IN，零回归。**
  - carotid（3 卷）：IN seam 1.47-1.71x → BN 0.51-0.54x，平均 seam -67.2%，L1 -23.2%。
  - muscle（1 卷）：IN 1.56x → BN 0.53x，seam -65.7%，L1 -24.8%。
  - phantom（1 卷）：IN 0.89x → BN 0.50x，seam -43.9%，L1 -6.6%。
  - phantom 上 IN seam=0.89x 已 <1.0，说明 IN 在 phantom 上本就没有产生伪缝
    （与 carotid/muscle 上 IN seam 1.47-1.71x 形成对比）。BN 将其从 0.89 压到 0.50，
    压的不是伪缝，更可能是过度平滑（抹平真实边界结构）。佐证：phantom L1 仅降 6.6%
    （carotid 降 23-26%），精度提升骤降，符合"没有伪缝可治、只是在平滑"的特征。
    此现象与 abs 部分文件回归（最差 -10pp）可能同源：BN 在缺少伪缝的简单信号上倾向
    过度平滑。原因待查。
- **判决**：BN 治缝效果在 carotid/muscle 上跨卷确凿（IN seam>1 的卷上 seam 和 L1 同步
  大幅改善）。phantom 上 IN seam<1（无伪缝），BN 仍压降 seam 但 L1 改善骤减，
  疑为过平滑。BN 在无伪缝信号上的行为待 codex 进一步分析。

### #2 前置判断与多卷诊断准备（2026-05-31，Carotid_012 现有数据）

- 问题：BN 已解决 #1 后，baseline 的空间不连续（约 2.8-3.0x label）是否仍传染到最终 pred？
- Carotid_012 frame1 结果：baseline x-boundary jump / label 为 2.87x（xz y=64 为 3.15x）；
  IN pred 为 1.64x（xz y=64 为 1.52x）；BN pred 为 0.50x（xz y=64 为 0.61x）。
- 视觉：同尺度 xz 图、db_min=-40 self-ref 压力显示、跨边界剖线均显示 IN 有明显 block，
  BN 基本无肉眼可见 patch 边界缝。
- 判决：在 Carotid_012/carotid 上，#2（数据 b 空间不连续）在 BN 后 pred 层面基本不是瓶颈，
  baseline 的 2.87x 不连续未显著传染到 pred，#2 在 carotid 上降为低优先级。
- 边界：此结论不能外推到 phantom/muscle。phantom 上 IN 无伪缝而 BN 仍压 seam，可能是好事也可能是过平滑；
  已新增 `rf_bn_multivolume_diagnostics.py`，等 proma 新 dense 到位后批量输出视觉图、seam、
  非边界 L1 和高频/纹理保真指标，统一区分“好的无缝”和“坏的过平滑”。

### 13 卷 BN 多卷诊断收官（2026-06-01，dense64 全类别）

- 数据：`dense64_index.csv` 共 13 卷，carotid 5 / muscle 4 / phantom 4。
  F 盘 WSL 9p 断挂后已重挂，F/H 两盘 raw patch 全部可读；每卷 256 个 HDF5。
  诊断汇总：`test_metrics/tiny_bn_random64_full1500/bn_multivolume_diagnostics_13.csv`。
- 方法：对每卷 IN vs BN dense64 no-overlap 拼图运行 `rf_bn_multivolume_diagnostics.py`，
  统一统计 seam、排除 patch 边界 ±3 voxel 的非边界 complex/abs improvement，
  以及 HF RMS/label、HF L1 improvement、detail retention、纹理对比度比。
- carotid（5 卷）：BN seam 0.496（0.483-0.501），非边界 complex improvement
  0.586（0.574-0.599），abs 0.554，HF RMS/label 0.472。BN 全面优于 IN，
  非边界 complex improvement 反升，判为非过平滑。
- muscle（4 卷）：BN seam 0.508（0.495-0.514），非边界 complex improvement
  0.592（0.581-0.608），abs 0.579，HF RMS/label 0.504。行为贴近 carotid，
  BN 全面优于 IN，判为非过平滑。
- phantom（4 卷）：BN seam 0.458（0.447-0.475），仍显著压低且整体优于 baseline；
  但细节指标一致低于 carotid 一档：非边界 complex 0.565（vs carotid 0.586）、
  abs 0.520（vs 0.554）、HF RMS/label 0.436（vs 0.472）。4 卷高度一致，
  定性为系统性轻微细节损失，而非严重过平滑。
- 机制假说（推测，未直接证实）：BN 使用训练集全局统计，phantom 作为最简单/低对比信号
  可能偏离全局分布最远，因此不如 IN 贴身，细节保真略低。内部对照：
  Phantom_088 是唯一 IN seam>1 的 phantom 卷，其 BN 非边界 complex=0.578，
  为 phantom 中最高，最接近 carotid。
- 判决：BN 方案定稿采用，缝根因 #1 收官。phantom 轻微细节损失是已知可接受 trade-off，
  当前不为此开新工程；若未来 phantom 重建质量成为瓶颈，再考虑 phantom 单独 IN 或混合归一化。
  #2（数据空间不连续）在 BN 后 pred 层面未显著传染，降为低优先，未来需要时再议 overlap/大块。

## unet_random64_full1500
- 数据配方:同 tiny_random64_full1500,使用 Data_cache_random64_full1500。
- 模型:UNetResidualRFNet,9,082,658 参数。
- 训练:batch_size=2 正常完成无 OOM;early stopping 于 epoch 42,
  best_epoch=27,best_val_l1=0.14698660228632193。
- Test overall:n=225,complex 3288.98 / 5614.82,improvement 45.34%,
  better 96.44%;abs 3022.98 / 5886.37,improvement 51.09%,better 97.33%。
- Per-category:carotid complex 43.77%/abs 51.79%;muscle 44.06%/49.13%;
  phantom 48.19%/52.36%。
- 判决:patch 级 improvement(complex 45.34% / abs 51.09%)均低于 Tiny
  (47.32% / 52.50%);边界跳变 1.53x 亦高于 Tiny 1.46x;全 volume L1 6703>6542。
  三类指标一致:UNet 未赢。唯一变量为网络结构。证伪"缝=感受野不够"假设。
- 关键现象:UNet val best(0.14699)略优于 Tiny(0.14754),但 patch test
  与拼图边界跳变均更差 => 再次印证"val/patch 指标不能判拼图质量"
  (claude.md 思维纪律第 3 条)。

## 动态范围线程 [2026-06]
- abs_weight sweep {0.1,0.5,1.0,2.0}:重加权无法修复压缩(slope 钉死 ~0.065、欠射只从 -0.84 挪到 -0.79);HF_ratio/detail_top10 的"提升"是细麻点纹理刷分,视觉证实更平;aw=0.1 视觉最忠实。commit 85ffd90
- SSIM 损失实验(lambda=1.0,3D 高斯 win=7):SSIM 项主导 loss(~5xL1)但 val_ssim 只到 ~0.21;首次撬动 slope(pooled 0.16->0.41)与 abs_std(0.29->0.70),欠射 -0.84->-0.72。commit d5a9544
- SSIM 反作弊验证:2D-hist slope 变陡且 pearson r 0.681->0.703(不降反升);误差 z-自相关与 baseline 几乎不变(无变白=无噪声注入);线性视觉为更亮非麻点。判定:真动态范围恢复。commit 66ab944
- SSIM 下加容量(WideDeepResidualRFNet 3.74M):p99 卷级 0.82、deep 0.95、abs_std 0.84 继续向 label 走,r 维持;但 test_ssim 0.221->0.222 几乎不动 -> 结构天花板是不可约 speckle、非容量。注:wide_final(ep35)比 best_by_val_ssim(ep13)动态范围更差 -> L1 随训练回拉压缩,选模用 val_ssim。commit 23732ad
- 卷级验证(Carotid_008):动态范围 0.27->0.81、deep 0.31->0.95,但 seam 0.50->1.19 回退(patch 级指标看不到)。commit 32191ef
- seam 诊断:x-jump profile 在 x=32/64/96 周期尖峰、wide>>BN-L1,符合深模型 ~8 voxel 边缘污染。z-band 验证:overlap(config2)/crop8(config3)把 seam 压回 <1、动态范围不掉,零重训。commit b804b96 / c9f5348
- full-volume crop8 流式(Carotid_008):seam x0.83/xz_y0.77 <1 无周期峰,动态范围全深度成立。commit f8ae843
- 跨组织推广(Carotid_073 / Muscle_056 / Phantom_036,halo 拼装 crop8):carotid 复现、muscle 优秀(p99 0.97),三类缝均 <1 无峰;phantom 深部过冲(deep p99 1.19)。commit b5d1f2f

### 人眼 3D 验证否决(2026-06,转 GAN)
- pred/label/BN-L1 导未压缩 .nii(各向异性 z=0.0362/x/y=0.2),3D Slicer 人眼验证。
- 发现:(a) phantom 线靶在 pred 丢失(label 有 pred 无,用户确认);(b) 组织 speckle 形态差(mush)。
- 判决:推翻"动态范围线程已收官"。回归(L1/SSIM)只改代理指标,达不到 DAS 真实散斑;根因=回归 -> 条件均值,散斑随机成分被平均、稀疏强点被对冲。
- 流程教训:此前仅凭聚合代理指标判"达标"、跳过人眼+标准指标,导致错误结论被写进 docs。今后质量结论必须过验证关卡。
- 转向:conditional GAN(对抗+保真),主攻真实散斑;点靶留后。
