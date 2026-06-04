# RF-to-Volume 项目:协作方式与决策逻辑交接

这份文档不重复 PROJECT_HANDOFF.md 的事实清单。它记录的是人类(项目负责人)
与 AI 助手之间行之有效的协作方式、已形成的思维纪律,以及关键决策背后的推理。
新接手的 AI 助手请先读这份,再读 PROJECT_HANDOFF.md 和 EXPERIMENTS_LOG.md。

## 三方协作结构
- 人类:决策者 + 唯一能看真实数据/图/环境的人 + 在 AI 和 codex 之间传递信息。
- AI 助手(你):参谋。做架构设计、实验设计、结果诊断、写给 codex 的精确指令、
  写给人类的判断。你看不到磁盘和 GPU,只能通过人类贴回来的真实产物了解状态。
- codex:执行者。在真实环境里写代码、跑实验、产出 CSV 和图。它不做战略判断,
  只执行明确指令。它不知道决策思路。

- 文件更新分工:AI 助手负责判断与给出"该写入的确切内容";
  codex 负责把内容写进 md/代码文件(codex 可直接访问项目文件);
  人类负责转发与确认。仅当改动极碎易错时,AI 直接出完整新版文件由人类替换。

## 这套协作成立的前提
- 人类必须把真实产物(CSV、图、日志、startup check)贴回来,你才能判断。
  人类转述会丢信息。越接近原始事实,判断越可靠。一个真实 bug(配置错配)
  就是靠人类贴真实 notebook 输出才发现的。
- 给 codex 的指令要精确到:建哪个文件、函数签名、怎么自测。并且让 codex
  "先贴代码/先空跑/先小规模验证,再实际执行"——不要一次写完一次跑完,
  否则多个 bug 缠在一起无法定位。
- 凡是生成大数据/重训这种高成本动作,先用小规模(pilot / val split /
  dry-run)验证管道正确,再放大。

## 已形成的思维纪律(最重要,务必遵守)
1. 一次只改一个变量。换 patch 尺寸时不同时换网络;否则结果无法归因。
   这条纪律救过项目多次。
2. 视觉判读会被显示参数(dB ref、aspect)欺骗。视觉和数值冲突时,
   以客观量化指标仲裁(例如 sliding-window 的"边界跳变指标"直接量化 block 缝,
   不受显示影响)。曾经人类和 AI 都被 dB 共享 ref 骗,以为模型退步,
   实际边界跳变指标证明模型更好。
3. patch 级 L1 对 block 缝几乎不敏感。不要用 patch 级 test improvement
   判断拼图质量。拼图质量只能看完整 volume 拼接 + 边界跳变。
   (2026-05 UNet 又一次验证:UNet val best 略优于 Tiny,patch test 与
   拼图边界跳变却双双更差。val/patch 与拼图质量脱节是稳定现象,不是偶然。)
4. 不被表象推着走。出问题先做便宜的诊断(读 CSV、跑评估),确认根因,
   再决定动作。曾怀疑"5 patch 配方过拟合",诊断后发现 train/test gap 只有
   9%,并非过拟合,及时收回了错误判断。
5. 配置错配是 Jupyter 的头号风险。训练前用 startup check 打印
   实际模型类名(type(model).__name__)+ 实验名/目录一致性 assert 拦截。
   每次正式实验必须 Kernel Restart & Run All,禁止单 cell 重跑。
6. 面对性能/报错/异常，先量化诊断定位瓶颈（微基准、配置 dump、资源监控），
   拿到数字再改，一次只改能归因的变量；改动多个文件后，
   开训/评估前必须确认主变量未被调试残留污染。
   （实例：I/O 优化时 multi-worker 在 WSL 大张量 IPC 下崩溃，经诊断回归
   num_workers=0；BN 结果出来后先做 3 项验证排除过度平滑才采信。）
7. 执行纪律（每次必守）：
   a) 实验输出开头打印时间戳 + 任务名；训练记每 epoch 耗时 + 总时长。
   b) 数字先报事实，异常/回归如实标出不淡化，判读交上游。
   c) 产出即 commit，一动作一 commit，消息写清，不混提交。
8. 质量结论必须过【验证关卡】,不得只凭聚合代理指标(slope/SSIM/abs_std/p99)。关卡=人眼 nii/3D-Slicer 签收(主:pred 与 label 同窗位并排,逐项看散斑形态/相干结构/点靶/幻觉/缝)+ 标准指标(speckle SNR≈1.91、gCNR、PSNR、点靶 FWHM,分 carotid/muscle/phantom 报)。人眼 fail 一律 fail,指标不能翻盘;codex 只产出 nii+指标+对比图,不判质量。血的教训:动态范围线程曾仅凭代理指标判"收官",人眼验证发现点靶丢失+散斑 mush,结论被推翻。

## 关键决策链(为什么这么走)
- Tiny 是强 baseline:Grouped/EarlyMix(更复杂但无下采样、感受野小)都没赢过
  Tiny。说明"更复杂≠更好",真正缺的是感受野/上下文。
- simu_point 分离:点靶任务(PSF/旁瓣/动态范围)与组织任务(speckle/结构)
  本质不同,不应混在同一 nonpoint 模型里。

## 缝根因(已分解,2026-05):多因叠加,InstanceNorm 是来源之一

baseline 边界跳变对照诊断给出第一条硬证据(Carotid_012 frame1,dense64):
- baseline/label = 2.99x(三边界均值)。baseline 未过网络,纯逐 patch
  输出即有近 3 倍跳变,说明缝不是网络独自产物。
- pred/label:Tiny 1.46x、UNet 1.53x,均 << baseline 2.99x。
  => 网络其实把输入/归一化/统计带来的跳变压掉一部分,既可能缓解也可能引入缝。
- UNet vs Tiny 共用同一 baseline(2.99x)。Tiny 压到 1.46、UNet 仅 1.53。
  => 大感受野本身没有额外压缝力,单纯换更强网络已排除。

【2026-05 缝根因完整分解(四轮验证收敛)】
缝不是单一根因,是多因叠加,逐层验证如下:
- 外层 per-patch scale(乘法归一化):非来源。统一 scale 重还原 baseline
  seam 2.99x 几乎不降(median 2.79x/mean 3.99x)。乘除抵消改不了底层。
- 网络内层 InstanceNorm3d per-patch 统计(track_running_stats=False,
  推理时每 patch 用自身空间统计):是来源之一,已坐实。
  推理时改用共享固定统计 + RMS 幅度校正(防压扁),
  pred seam 从 1.4618x 降到 1.2907x(约12%真实改善;
  未校正的 0.49x 主要是幅度压扁假象,不可信)。
- 数据 b 在 patch 边界的空间不连续:剩余缝(1.29x→1.0x 的差距)来源,
  InstanceNorm 共享治不到,需 overlap/大块从空间上处理。

方向(更新排序):
  1. 【已完成 2026-05-29】InstanceNorm3d → BatchNorm3d 重训 (tiny_bn_random64_full1500)。
     **注意：此前文档建议"InstanceNorm 改 track_running_stats=True"是错误方案。**
     InstanceNorm 改 track_running_stats 仍为 instance-level 归一化，推理时每 patch
     仍用自身统计；正解是换成 BatchNorm3d(affine=True, track_running_stats=True)，
     训练累积 running stats、推理时全局共享，才是"跨 patch 共享归一化基准"。
     此坑已踩，修正留此条以防重蹈。
     结果：seam 1.46→0.535x（-63%），patch complex impr 47.32%→56.85%（+9.5pp），
     full-volume L1 -20%，边界/非边界 L1 等比例降。验证排除过度平滑——
     IN 在个别位置引入的 45x/9.7x 假台阶被 BN 压回 ~0.85x 跟随 label 结构；
     个别位置（z=512 step/label=0.33x）有轻微过平滑但无损总账。
     判决：缝根因 #1（InstanceNorm per-patch 统计）已确认并根除。
  2. 数据 b 空间不连续仍存在（baseline seam 仍约 2.8-3.0x），但后续
     13 卷诊断显示它在 BN 后的 pred 层面没有成为主瓶颈，已降为低优先级。
     overlap+Hann 或大块/整卷推理只作为未来需要进一步压残余空间不连续时的候选。
     大块直接推理会因尺寸失配崩坏（BatchNorm 统计也随尺寸变），大块必须配合尺寸适配重训。
  3. 已排除：统一外层 scale（无效）、单纯换更强网络（无额外压缝力）、
     InstanceNorm 改 track_running_stats（无效，是错误方案）。

关键方法论遗产:缝是多因叠加,单点干预只压一部分;每次只改一个变量、
幅度校正剥离压扁假象、seam_metric 量化仲裁 —— 是定位多因的关键纪律。

### 2026-05-31 前置判断:#2 在 carotid/Carotid_012 上降级,但不可外推到 phantom

问题:BN 根除 #1 后,baseline 的 2.8-3.0x 空间不连续到底还有多少传染到最终 pred?
若 BN pred 已基本无缝,#2(overlap/大块)不应立刻成为主投入。

Carotid_012 frame1 现有拼图诊断:
- baseline x-boundary jump / label:full-volume 2.87x,xz y=64 为 3.15x。
- IN pred / label:full-volume 1.64x,xz y=64 为 1.52x,视觉仍有明显 block。
- BN pred / label:full-volume 0.50x,xz y=64 为 0.61x,视觉图和 db_min=-40 self-ref 压力显示均基本无 block 缝。
- 结论:在 carotid 的 Carotid_012 上,#2(数据 b 空间不连续)在 BN 后 pred 层面基本不再是瓶颈;
  baseline 的 2.87x 不连续未显著传染到最终 pred,#2 在 carotid 上降为低优先级。

判读边界:
- 这个结论只覆盖 Carotid_012/carotid,不能外推到 phantom/muscle。
- phantom 上 IN seam 已 <1(无明显伪缝),BN 继续压到 ~0.50 且 L1 改善骤减,可能是"好地消除残余不连续",
  也可能是"坏地抹平真实结构"。因此 #2 残余空间不连续与 phantom/abs 过平滑疑点是同一问题的两面,
  需要多卷视觉 + seam + 非边界结构保真指标统一判读。

### BN 多卷诊断方案(等待 proma 新 dense 到位后执行)

执行脚本:`rf_bn_multivolume_diagnostics.py`。每个 dense volume 输入 IN 拼图目录和 BN 拼图目录,
输出统一 CSV 与三类图:
1. `IN_vs_BN_full_xz_y64_boundaries.png`:同 dB ref 的 baseline/pred/label 整张 xz 图,红线标 x=32/64/96。
2. `IN_vs_BN_pred_selfref_dbmin40_xz_y64.png`:pred-only 压力显示,每个模型 self-ref,db_min=-40,放大残余 block。
3. `IN_vs_BN_x_boundary_profiles_y64.png`:跨 x=32/64/96 的平均剖线,看边界台阶是否跟 patch boundary 对齐。

量化指标:
- seam 指标:`x_boundary_pred_over_label = mean(|abs_pred[x]-abs_pred[x-1]|) / mean(|abs_label[x]-abs_label[x-1]|)`。
  <1 表示 pred 边界跳变低于 label 自然跳变;>1 表示仍有伪缝风险。
- 非边界 L1:`nonboundary_complex_improvement`、`nonboundary_abs_improvement`,排除所有 patch 边界 ±3 voxel。
  过平滑若是真的,通常不应只看 seam 下降,还会伴随非边界 abs/complex 改善弱、停滞或回归。
- 非边界高频/纹理保真:`pred_hf_rms_over_label`、`detail_retention_top10`、
  `pred_nonboundary_abs_std_over_label`、`pred_hf_improvement_vs_baseline`。高频显著低于 label 只是疑点,
  需结合非边界 L1 和视觉判读;若 seam 降而非边界 L1 不升、detail retention 很低、纹理对比度显著低,
  才判为过平滑。

判读标准:
- 好的无缝:BN seam 明显低于 IN 且接近/低于 label,视觉无 block,非边界 complex/abs L1 同步改善,
  高频虽可低于 label 但结构边界仍随 label、`pred_hf_improvement_vs_baseline` 为正。
- 坏的过平滑:原本 IN seam 已 ≤1 的 volume 上,BN 继续压 seam 但 L1 改善很小或 abs 回归,
  self-ref/剖线显示真实结构被抹平,并且非边界高频/对比度/Top10 detail retention 明显低于 label 与 IN。
- phantom 特别关注:若 phantom BN seam 低但高频/对比度显著低于 label 且非边界 abs 不改善,过平滑成立;
  若高频接近 label、非边界 L1 改善且视觉保结构,则是真无缝。

### 2026-06-01 13卷多卷诊断收官:BN 定稿采用,#1 收官

数据:13 卷 dense64,carotid 5 卷 / muscle 4 卷 / phantom 4 卷。
`dense64_index.csv` 的 source_file 已核对;F 盘 WSL 9p 断挂后已重挂,
13 卷 raw patch 当前均可读,每卷 256 个 HDF5。诊断输出:
`test_metrics/tiny_bn_random64_full1500/bn_multivolume_diagnostics_13.csv`。

客观汇总:
- carotid(5 卷):BN seam 均值 0.496(0.483-0.501),非边界 complex improvement
  0.586(0.574-0.599),abs improvement 0.554,HF RMS/label 0.472。
  BN 相比 IN 全面压缝,非边界 complex improvement 反升,判为非过平滑。
- muscle(4 卷):BN seam 均值 0.508(0.495-0.514),非边界 complex improvement
  0.592(0.581-0.608),abs improvement 0.579,HF RMS/label 0.504。
  行为贴近 carotid,BN 全面优于 IN,判为非过平滑。
- phantom(4 卷):BN seam 均值 0.458(0.447-0.475),仍显著压低;
  非边界 complex improvement 0.565(vs carotid 0.586),abs improvement 0.520(vs 0.554),
  HF RMS/label 0.436(vs 0.472)。4 卷高度一致,说明是系统性轻微细节损失,
  不是偶然异常;但非断崖式,且整体仍优于 baseline,定性为可接受 trade-off,
  非严重过平滑、非 deal-breaker。

机制假说(推测,未直接证实):BatchNorm3d 用训练集全局统计归一化,phantom 是更简单/
低对比信号,可能偏离全局分布最远,因此 BN 对 phantom 不如 IN "贴身",
细节保真略低。内部对照支持:Phantom_088 是唯一 IN seam>1 的 phantom 卷,
更像复杂信号;其 BN 非边界 complex improvement=0.578,为 phantom 中最高,
最接近 carotid 区间。

定论:
- BN 方案定稿采用。`tiny_bn_random64_full1500` 作为当前 RF-to-volume 主 baseline。
- 缝根因 #1(InstanceNorm per-patch 统计)已根除:换 BatchNorm3d 后,
  13 卷三类 seam 全面压低(IN 约 0.82-2.07x -> BN 0.45-0.51x);
  carotid/muscle 重建同步提升且非过平滑,phantom 轻微细节损失是已知可接受代价。
- 缝根因 #2(数据空间不连续,baseline 约 2.87x)在 BN 后 pred 层面未显著传染
  (carotid 已验证),降级为低优先。未来若它重新成为瓶颈,再考虑 overlap/Hann 或大块重训。
- 不为 phantom 轻微 trade-off 开新工程;若未来 phantom 重建质量成为瓶颈,
  候选是 phantom 单独用 IN 或混合归一化。

方法论收官:多因分解、廉价诊断先行、数值+视觉+多卷交叉仲裁、兴奋时先验证。

## 动态范围线程(回归范式已否决)—诊断链与转向 conditional GAN [2026-06 修正]

### 修正判决
回归范式(L1/SSIM 拟合 DAS)已到天花板:数学上逼近条件均值,产不出随机散斑。WideDeep+SSIM+crop8 只改善了 slope/p99/abs_std/SSIM 等代理指标;NIfTI + 3D Slicer 人眼验证下失败:phantom 线靶丢失(label 有 pred 无,已确认),组织 speckle 形态差(mush)。此前"动态范围线程已收官/交付配方"结论作废,本节作为负结果和诊断纪律保留。目标转 conditional GAN:对抗出真实散斑 + 保真锁结构防幻觉。主攻组织+仿体真实散斑;点靶简单留后;simu_point 暂不进训练;扩散列 future work(3D+8GB 吃不消)。

### 诊断链(作为"廉价诊断先于大投入"的范例)
L1 残差 -> 确诊系统性幅度欠射/动态范围压缩(增益主导 perp/par~0.18、信号成比例 corr~0.88、跨难度组均匀 -> 是损失导致的回归到均值,不是不可约) -> 廉价诊断先否掉一批便宜但错的杠杆:HF 损失(pred_hf_improvement_vs_baseline 已 +0.67,缺的高频是不可约 speckle)、定向加权(误差分布窄、无集中尾)、重采样(三类已均衡)、L1 下加容量(回归到均值是损失形状问题,容量治不了,也解释了 UNet 当年没赢) -> abs_weight 重加权治不动(slope 钉死 ~0.065;HF/detail 被细麻点纹理"刷分",视觉证实是更平 -> 数被作弊) -> SSIM 目标(协方差项要求与 label 相关,天然抗噪声刷分)首次撬动 slope(0.065->0.167)、abs_std(0.29->0.70) -> 反作弊三验确认是真的:2D-hist 里 slope 变陡同时 pearson r 不降反升(灌噪声只会让 r 掉)、误差自相关没变白、线性视觉是更亮而非麻点 -> SSIM 下加容量(WideDeep 3.7M)继续推进(卷级 p99 0.82、deep 0.95),但 val_ssim 卡在 ~0.21 -> 结构匹配的天花板不是容量、而是 speckle 的不可约成分 -> 卷级验证抓出 seam 回退(patch 级指标对缝盲) -> seam = 深模型边缘 zero-padding 污染(随深度增大、patch-grid 周期) -> crop8/overlap/halo 在推理协议层面修掉,零重训零动态范围代价 -> full-volume + 多组织代理指标一度看似通过 -> NIfTI/3D Slicer 人眼验证否决(phantom 线靶丢失、组织 speckle mush),回归范式归类为负结果。

### 本线程结晶出的思维纪律(通用)
1. 指标可被刷分,必须用"不可作弊指标 + 视觉"交叉仲裁。判"真跟上 vs 灌噪声":slope 变陡的同时 pearson r 是否同时上升(r 掉=噪声虚高)。
2. dB 视觉会在动态范围上骗人(对数压缩),判动态范围要看线性 / slope / p99,别只看 dB 图。
3. patch 级指标对拼接缝是盲的。任何模型在宣布"可用"前,必须做 full-volume 验证。
4. 选模标准必须对齐目标(realism 下用 val_ssim,不是 val_l1;训练越久 L1 越把压缩拉回来,wide_final ep35 在动态范围上反而不如 ep13)。
5. 代理指标上的胜利,要在"交付尺度 + 数据分布广度"(整卷 + 多组织)上验证后才算通用;OOD 区域可能翻向(phantom 深部从欠射翻成过冲)。
6. 边缘污染随模型深度增大;部署协议层面的修法(crop/halo)优于回炉重训。
7. (元)整条链体现:廉价诊断先杀掉便宜但错的杠杆再上贵的;"兴奋时先验证"两次救场(都是视觉戳穿了被刷分的数)。

## 各向异性是这个项目的物理底色
RCA-OPW 网格:轴向 0.0362mm、横向/纵向 0.2mm,比例 1:5.5。
- 显示:imshow aspect 必须按物理间距设(xz/zy 用 0.181,xy 用 1.0)。
- 网络下采样:x/y 可下采样,z 要保守(否则丢轴向细节)。这是 3D-ARCGAN
  论文的核心,RF 线的 UNet 也遵循此原则。

## 与人类协作的偏好
- 人类对话额度有限(token 成本)。沟通要"重而少":一次把方案给全,减少来回。
  人类只贴结果摘要(startup check + summary CSV),不贴整文件。
- 人类会质疑 AI 的判断,这是好事。不一致时用数据仲裁,不靠谁声音大。
- 给 codex 的指令尽量让人类一次复制转发,不要拆碎。
