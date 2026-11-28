# 当前源码运行时

实施依据和逐项状态分别见 [方案](plan_zh.md) 与 [验收记录](execution_zh.md)。本轮入口采用五区块 JSON；原来的扁平 CLI 参数已被子命令替换。旧 Executor、ConfigParser、自定义 Batch 和历史加载路径已在阶段五删除；待迁移模型源码仍保留，但不会回退到另一套运行时。

阶段一、二的 184 项记录保留为历史基线。阶段三修复预测输入和目标时间轴，阶段四接入五类任务代表组合及统一调参。当前范围、依赖与验证级别见 [模型迁移表](models_zh.md)，新增实施记录见 [阶段三、四记录](phases34_execution_zh.md) 和 [验收报告](phases34_acceptance_zh.md)。

当前正在继续完成剩余模型迁移与主线缺口，最新提交、修复说明和验收范围见 [主线剩余问题执行记录](remaining-work_execution_zh.md)。历史阶段验收通过不等同于全模型迁移完成；当前支持范围以迁移 CSV 为准。

公共点数据现按真实 geo_id 和 UTC 时间对齐，支持节点分块、时间优先和完全乱序的 .dyna，拒绝重复/缺失/未知节点记录。点原始缓存切换到新命名空间，元数据记录 node_ids 与 point_alignment；缺少新身份说明的旧点检查点需重新训练，不自动补声明，详见 [节点身份修复](point-identity_zh.md)。

HIEST 已接入重叠区域、逐样本三级图和显式重建/余弦辅助训练；用 `--model HIEST` 生成 hiest.json 后执行统一训练、独立评估和续训。自动双连通分区覆盖关节点、桥和孤立节点，也可通过 data.graph.region_groups 指定真实 ID 组。普通 forward 只返回预测，评估不执行最终辅助图或使用辅助损失，输出支持全部预测步和目标通道；配置、数学修订及验收见 [HIEST 说明](hiest_zh.md)。

OD 原始数据和多空间轴指标已补齐公共组件，支持起点/终点身份、原始正值掩码、训练区间归一化、历史与已知未来协变量分离、缓存及来源版本校验，见 [OD 数据说明](od-data_zh.md)。GEML、CSTN 均已接入。

STDN 已接入局部流量门控和周期偏移注意力，用 `--model STDN` 生成 stdn.json，使用相同的 train/evaluate/resume 生命周期。`.grid` volume 与两列 `.gridod` 流量按实际节点和时间对齐，局部块及历史向量按需切片，全部输入在预测起点之前；多步头从同一预测起点并行输出。两个 scaler 仅拟合原始训练区间，原始缓存分别复用。配置、时间错位修复及验收见 [STDN 说明](stdn_zh.md)。

ESG 已接入演化图和多尺度时间卷积，点/网格数据共用标准预测输入。`--model ESG` 生成 esg.json 后可使用相同生命周期；data.graph.history_steps/history_channels 控制原始训练前缀及目标通道，默认全部训练区间，不依赖滑窗数量。图历史注册为 buffer，单节点 BN 有有限回退，零 hop 不创建无效图分支；动态图公式和有符号传播保持，去掉死层及节点对重复拼接。配置和验收见 [ESG 说明](esg_zh.md)。

GEML 可用 `--model GEML` 生成单流量 OD 示例，再对 geml.json 使用相同的训练、独立评估、续训命令。模型保留地理/语义图和周期记忆，纯 forward 无标签且不计算辅助头；训练 step 使用原尺度 OD 与行/列边际复合目标，验证只用主 OD MSE。p_interval、embed_dim 和 training.loss 的三项权重可显式设置；周期记忆及原辅助目标错误已修复。具体语义与能力见 [GEML 说明](geml_zh.md)。

CSTN 可用 `--model CSTN` 生成网格 OD 示例，使用相同的生命周期命令。两条 OD 轴根据 .geo 网格位置转换/恢复；全局相关按区域逐列归一化，不混入其他样本。外部特征按真实宽度编码并广播，预测只推进已知未来列，未知天气使用最后观测。data.graph.name=none，无需 .rel。网络保留 LibCity 的 LSC/TEC/GCC 结构，修订及验收见 [CSTN 说明](cstn_zh.md)。

DSAN 可用 `--model DSAN` 生成连续网格示例，使用 `grid_window` 与无静态图配置；原始历史掩码在归一化前计算。全局/局部与空间/时间注意力保持明确样本轴，所有堆叠逐层传递；默认训练为公共 masked_rmse，支持梯度累积和精确续训。采用历史种子并行预测，不读取目标或未来观测；参数、长预测窗扩展和算法修订见 [DSAN 说明](dsan_zh.md)。

GSNet 已接入原生网格/多图时空网络、风险分档目标和区域排名评估；用 `--model GSNet` 生成 gsnet.json，再执行相同的训练/独立评估/续训命令。`.geo` 可只覆盖部分图节点，映射到完整网格；未来上下文只读取第一真实预测时刻的已知信息，多步目标全部监督。默认 risk_cpt 最近 3 帧加前 4 周同刻；示例显式缩短周期。配置、修订及实际验收见 [GSNet 说明](gsnet_zh.md)，数据/目标/指标公式见 [风险组件说明](risk-components_zh.md)。

道路表征的 ChebConv/GAT/GeomGCN 使用具名 `road_reconstruction` 协议，任务默认配置与模型预设分开加载；预设仍不能静默改写协议字段。协议名称及相对于默认边采样协议的差异会写入日志和 runtime_info.json。生成示例可用 `tools/make_runtime_fixture.py --directory <目录> --model ChebConv`、`--model GAT` 或 `--model GeomGCN`；它们重建静态节点属性并导出全节点嵌入，使用全图批次与互斥监督节点集合，不将此指标解释为下游分类精度。GeomGCN 的道路边/自环分区以 PyTorch 聚合实现，不需要 DGL；训练、独立评估与完整续训均使用公共入口。它保留 LibCity 的两分区重建变体，不包含原论文的几何邻域构建；合并维度与有意修复见模型说明。

STResNet/ACFM 的原子网格示例可用 `tools/make_runtime_fixture.py --directory <目录> --model STResNet` 或 `--model ACFM` 创建，再对生成的 stresnet.json/acfm.json 执行相同的 train/evaluate/resume 命令。`grid_cpt` 协议默认单步、最近 3 个时间片和前一天同刻；data.sampling 可显式配置周周期，output_window 可显式增大。`.geo` 需包含完整矩形的 row_id/column_id；`.grid` 行按时间/行列对齐，输出沿原 geo_id 顺序排列。历史外部观测与未来已知特征分开，只有日历及 known_future_columns 会进入预测时刻输入；不能把实际未来天气或观测当已知。默认缓存可复用，.grid 修改会使缓存失效并拒绝旧状态续训。详见 [模型说明](models_zh.md)。

CONVGCN/MultiSTGCnet 示例可用 `--model CONVGCN` / `--model MultiSTGCnet` 生成，使用 convgcn.json/multistgcnet.json 执行同样的训练、评估和续训命令。它们是点图模型，输入为 .geo/.rel/.dyna。CONVGCN 的 seasonal_windows 参数 period_steps/trend_steps 按原始时间片计数；MultiSTGCnet 的 frame_cpt 按最近帧/日/周采样，默认从 .rel 距离构建三组支持，不用高斯相似度代替距离。协议名、实际窗口数/跨度与距离边界都会进入运行摘要。详细参数与有意修复见 [模型说明](models_zh.md)。

DKFN 示例可用 `tools/make_runtime_fixture.py --directory <目录> --model DKFN` 生成，再对 dkfn.json 使用相同的 train/evaluate/resume 命令。它保留图/自身双循环流，方差按样本计算，避免其他 batch 样本改变预测；零方差使用有限回退。多步滚动只读取已声明的未来特征，未知观测沿用最后值，不读取 y。默认 RMSprop，hops/clamp_adjacency/threshold 位于 model.params；融合系数与有意数值变化见 [模型说明](models_zh.md)。

ResLSTM 示例用 `--model ResLSTM` 生成，对 reslstm.json 执行相同训练/评估/续训命令。至少提供两列原始流量，flow_indices 可选择对应列；历史日历/.ext 单独作为全网外部信息，不读取未来真实值。reslstm_windows 默认每段 5 帧、服务偏移 72/360 帧，发布夹具有足够长的时间轴。季节窗口之间允许停运间隔，但每段及近期到目标须连续，策略写入摘要；LSTM 沿节点轴运行，batch 样本互相独立。详见 [模型说明](models_zh.md)。

STSGCN 示例用 `tools/make_runtime_fixture.py --directory <目录> --model STSGCN` 生成，对 stsgcn.json 使用相同的 train/evaluate/resume 命令。示例启用图掩码，不需要 retain_graph。module_type 可选 individual/sharing，filters 的每个块内部宽度须相同，历史长度必须大于块数的两倍；act_type 可选 GLU/relu。默认图为 .rel 距离倒数图，训练为原尺度 Huber，阈值在 training.loss.delta 设置；需要排除原始零目标时显式选择 masked_huber。模型推理只读取历史 X，完整配置、有意修复和边界见 [模型说明](models_zh.md)。

STPGCN 示例用 `--model STPGCN` 生成，对 stpgcn.json 使用相同 train/evaluate/resume 命令。C/L/d 设置通道/层数/位置嵌入，alpha 是有向空间跳数上限，beta 是当前时段之前的历史范围；两者允许为零。日历从原子真实 UTC 时间戳生成独立 int64 history_features，不需要把星期、时刻加入数值 X。采样间隔须整除一天，按午夜计算时段。全部关系、卷积与选择层均参与训练并写入检查点；高斯关系采用数学等价的空间/时间分解，不增加专用训练循环。语义修订、范围与限制见 [模型说明](models_zh.md)。

DGCN 示例用 `--model DGCN` 生成，按 dgcn.json 执行相同 train/evaluate/resume 命令。具名 dgcn_cpt 默认取最近 2 小时、前一天和前两周的各 12 步，总共 60 帧；小时级小例子显式改为一步、共 5 帧。c_out/K/Kt/attention_heads/dropout 控制动态图通道、Chebyshev 阶数、奇数时间核、注意力头数与丢弃率，周期类别由 data.sampling 声明。模型按布局恢复每类内远到近顺序，粗周期输入轴已纠正。训练 MSE 默认包含原始零目标；边界、语义变化及当前公共 CPT 数据频率约束见 [模型说明](models_zh.md)。

HGCN 示例用 `--model HGCN` 生成，对 hgcn.json 使用相同生命周期命令。data.graph.cluster_nodes=0 自动选 min(10,N) 个区域；示例设为 2 并实际运行固定种子的静态路网谱聚类，也可用 cluster_labels 按 .geo 顺序给出分区。区域输入使用所有组员之和，图与分区是只读元信息，改变分区设置可复用原始缓存。temporal_kernels 默认 [3,2]，要求历史至少 10 帧；缩短核可运行更短历史。默认保留双层图与多步/多目标输出，全部训练/评估/续训使用公共入口。区域图的原高斯规则、退化处理和参数说明见 [模型说明](models_zh.md)。

GTS 示例用 `--model GTS` 生成，对 gts.json 使用同样的 train/evaluate/resume 命令。data.graph.history_channel 选择历史观测列，history_steps=0 使用完整原始训练区间，正数取最近若干训练时间点；所选序列须连续且至少有 2*kernel_size-1 帧。图特征不依赖窗口展开、padding 或 batch_size，不读取验证/测试值。模型结构位于 model.params，温度、课程学习与图正则截止位于 training.strategy；示例只在第一轮加图正则。训练保留 Gumbel 随机采图，评估使用确定图且每次完整评估只计算一次，forward 不读取 y。参数、原实现语义修订及规模限制见 [模型说明](models_zh.md)。

RGSL 示例用 `--model RGSL` 生成，对 rgsl.json 使用同样的生命周期命令。model.params 控制 rnn_units/num_layers/embed_dim/cheb_k，training.strategy.temperature 控制原作者采图直通梯度的尺度（默认 1）。训练显式随机采图，预测/评估使用确定图；各轮模型、优化器、调度器和采图 RNG 可以完整续训。静态图和掩码作为 buffer，每次前向只准备一次学习基底与各层节点参数；高阶静态图递推、零图/零嵌入和小通道边界已修复。默认 masked_mae 排除原始零目标，具体计算保留与修订见 [模型说明](models_zh.md)。

CCRNN 示例用 `--model CCRNN` 生成，再对 ccrnn.json 使用同样的训练/独立评估/续训命令。data.graph 使用 svd_gaussian，svd_rank 控制保留分量，svd_channels 可选择原始历史观测，normalization 可选 randomwalk/laplacian；图只从原始训练区间生成，.rel 无需提供。n_dim 自动限制到节点数，n_gconv_layers 可变；课程开关和衰减放在 training.strategy。默认 masked_rmse 包含零目标，按整组累积和完整验证集计算 RMSE，未使用未来目标作推理输入。两种归一化均完成精确续训；配置及有意修订见 [模型说明](models_zh.md)。

DMSTGCN 示例用 `--model DMSTGCN` 生成，对 dmstgcn.json 使用同样的生命周期命令。夹具包含两个真实观测列；data.features.primary_index/auxiliary_index 选择主/辅助历史观测，目标外的选中列单独用训练区间拟合归一化器。data.graph.name=dynamic，.rel 可省略。日历通过独立 int64 history_features 传入，按首个真实历史时间取时段，采样间隔需整除一天。模型自动推导感受野、skip 头和归一化尺寸，支持多目标及 batch/layer 归一化；原无效末层计算已清理。详细参数及修订见 [模型说明](models_zh.md)。

## GRU 完整例子

从源码根目录运行，数据、缓存和输出都可放在仓库外：

```powershell
python -m pip install -e '.[dev]'
python tools/make_runtime_fixture.py --directory ../runtime-demo
python run_model.py train --config ../runtime-demo/gru.json --set runtime.exp_id=gru-training
python run_model.py evaluate --config ../runtime-demo/gru.json --checkpoint ../runtime-demo/runs/gru-training/checkpoints/best.pt --set runtime.exp_id=gru-evaluation
python run_model.py train --config ../runtime-demo/gru.json --resume ../runtime-demo/runs/gru-training/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=gru-resumed
```

Python 使用同一入口：

```python
from libcity.pipeline.experiment import run_experiment
result = run_experiment('train', '../runtime-demo/gru.json',
                        overrides={'runtime': {'exp_id': 'python-training'}})
print(result.metrics)
print(result.artifacts['best'])
```

`data.data_dir` 是包含数据集子目录的根目录；`data.cache_dir` 与 `runtime.output_dir` 独立设置。JSON 中的相对路径相对于该 JSON 文件，`--set` 和直接传入 Python 字典中的相对路径相对于当前目录。默认值来自包资源。原子数据的 config.json 只提供数据文件与字段元信息，不能覆盖任务的窗口、切分和评估协议。

## 训练与产物

神经模型的 `best.pt` 仍是可单文件复制的安全 state_dict，附带输入说明元信息。独立评估在加载参数前核对模型、归一化、节点身份及输入布局，避免相同形状下静默错用数据。缺少说明的旧权重可从通过完整校验的 `last.pt` 重新导出，已完成 epoch 时无需额外更新；详见 [权重输入检查](inference-weights_zh.md)。

神经网络训练实验写入 `<output_dir>/<exp_id>/`，含 resolved_config.json、runtime_info.json、run.log、metrics.jsonl、checkpoints/best.pt、checkpoints/last.pt 和 eval/ 下的 JSON/CSV 指标及可选预测文件。`evaluation.save_predictions=false` 关闭预测导出。TensorBoard 仅在支持它的训练且 `runtime.tensorboard=true` 时创建；独立评估不创建优化器、训练记录器或 TensorBoard。

基础安装不要求 TensorBoard。启用该后端时安装 `pip install -e '.[tensorboard]'`；dev 组为可选记录器的验收保留该依赖。Trafformer 单独需要 `pip install -e '.[transformer]'`。

`best.pt` 是纯模型 state_dict。`last.pt` 保存最近完整轮次的模型、优化器、调度器、AMP、TrainState、早停、最佳权重、Python/NumPy/Torch 随机状态和训练 DataLoader 随机状态，全部使用 weights_only 安全加载。续训可以延长普通调度器的 epochs；按总更新数定义的 OneCycle/WarmupCosine 必须保持原定 horizon。数据选择、损失、优化器和累积设置不能在续训时悄悄改变。

last.pt 格式版本为 3，还校验只读数据元信息、采样布局、归一化状态与原子来源文件版本。来源在训练上下文建立时记录文件名、大小和纳秒修改时间，不哈希文件内容；数值或图文件发生版本变化时，续训在载入模型/优化器状态前拒绝，避免沿用另一份数据的最佳指标。数据目录内的名字使用相对路径，保留文件元信息后可移动目录；缓存和运行产物不作为原子来源。文件元信息检查是保守判定，单独改变修改时间也会拒绝，并非内容相等证明。旧格式缺少来源记录，需重新训练；best.pt 的独立评估不受此续训格式升级影响。ASTGNN 的两阶段计划同样必须保持一致。旧 `.m/.tar` 和本轮开发中较早的 runtime 格式不自动迁移。换成其他模型时，CLI 的 model.name 选择新预设并丢弃用户 JSON 中前一个模型的结构参数；其余显式实验参数仍按优先级保留。

`training.optimizer.groups` 接受模块前缀、lr、weight_decay 和 freeze。调度器显式设置 interval=step/epoch/metric。梯度累积按真实有效目标数归一化，并在每轮最后一个不足累积组的尾批完成更新；只有实际优化器更新才步进 step 调度器。默认循环支持最小化或最大化指定监控指标、早停、CPU/CUDA/MPS 设备选择和 AMP。CPU AMP 使用 bfloat16，并实际探测 oneDNN 卷积反向能力；后端不支持时使用原生反向内核，退出后恢复设置，前向仍保留 AMP。硬件覆盖以各批次记录为准；阶段五已有代表模型 CUDA 记录，MPS 尚未认证。

RMSE 等非线性目标接入默认循环时，需要声明可加统计的汇总与梯度归一化规则；公共组件现已预留 aggregate/gradient_divisor 两个损失方法。CCRNN 所需的 RootMeanSquaredLoss、整组 RMSE 梯度与跨批评估说明见 [专项说明](root-loss_zh.md)。这是底层扩展能力，模型可用性仍以迁移清单和各模型配置为准。

## 预测输入与目标时间轴

交通适配现在记录 `split.semantics=target_time_v1`。按原始时间点确定边界，只有目标窗口完整属于该区间的样本才能进入对应集合；丢弃跨边界目标窗口，历史可使用切分点前的已知观测。归一化只拟合原始训练时间区间，验证和测试不 padding。除默认比例外，可在 data.split 中同时提供 train_end/val_end（ISO 时间戳，右侧区间从该时刻开始）。真实 METR-LA 运行记录见 [报告](metr-la-phase3.json)。

批次的 X 只包含历史；y 是归一化后的预测目标，y_raw 是原始目标。损失与评估用 y_raw 判定零值掩码，不经过归一化再还原。future_features 只含日历和明确声明的已知未来协变量。例如 `data.features={"load_external":true,"known_future_columns":["holiday"]}` 允许从 .ext 读取 holiday 的未来值，其余外部观测只保留历史。GRU 的不可提前获取的辅助观测沿用最后历史值，不读取真实未来；Teacher Forcing 只在显式训练步使用 y。

旧展开滑窗缓存不再使用。`data.cache=true` 将未归一化的原始时序、已知未来特征与图缓存到独立 `cache_dir/traffic-raw-v1/`；命中后不再解析原子数据。特征/图配置隔离缓存；源文件路径、大小或修改时间变化时重建，不做源文件内容哈希。每次实验仍按当前配置重新切分、仅用训练区间拟合归一化并按需生成窗口，因此改变窗口、切分、归一化或 batch_size 可复用同一原始缓存。`runtime_info.json` 的 split.cache 记录 hit/miss/disabled 及文件位置；损坏缓存会重建，写入采用临时文件与原子替换。37 个特殊 Dataset 与重型变换缓存的深度整理仍延后。阶段一、二按旧切分生成的 last.pt 不自动续到新语义；数据/采样元信息检查会拒绝不一致状态。

ASTGNN 使用 data.sampling.name=cpt 及只读采样布局，实际历史长度与请求窗口的差异进入运行摘要。data.graph 控制二值/加权/距离图、对称化和自环，模型预设不会悄悄更改统一切分、窗口或指标协议。

公共 CPT 的小时/日/周位移分别按 3600、86400、604800 秒与采样间隔精确换算，interval_period 和 interval_trend 分别乘在日、周周期上。只检查实际启用的类别：如 7 分钟采样可使用完整周周期，2 小时采样可使用日周期；无法精确表示的启用周期在准备数据时明确拒绝。output_window 不能超过最短启用位移，保证每段历史的末帧早于目标；关闭小时分支后不再受每小时点数的无关限制。整小时约数的既有配置保持原索引/元信息，旧检查点仍可续训；修正了错误周期的配置，其 max_position 改变会触发已有安全续训检查。原始缓存仍可复用。具体前后实测见 [CPT CLI](cpt-cli-validation.json)。

## 无训练权重的地图匹配

```powershell
python tools/make_runtime_fixture.py --directory ../map-demo --task map_matching
python run_model.py evaluate --config ../map-demo/hmmm.json --set runtime.exp_id=map-evaluation
```

HMMM 直接评估，training 区块为空；不需要 checkpoint，不支持 train/resume。这里的轻量原子适配要求双端点 LineString 路段及每条轨迹的真值路线，不读取旧 pickle 缓存。预测保存为 predictions.json；metrics.json/csv 为逐轨迹指标的宏平均，metrics_by_trajectory.json 保留明细。它与交通模型共用实验目录与结构化结果，未强行套用窗口、损失或优化器配置。

## 轨迹、ETA 与道路表征

生成器可为五类任务生成小型原子示例。以下路径均可放到仓库外：

```powershell
python tools/make_runtime_fixture.py --directory ../trajectory-demo --task traj_loc_pred
python run_model.py train --config ../trajectory-demo/rnn.json --set runtime.exp_id=train
python run_model.py evaluate --config ../trajectory-demo/rnn.json --checkpoint ../trajectory-demo/runs/train/checkpoints/best.pt --set runtime.exp_id=eval
python tools/make_runtime_fixture.py --directory ../eta-demo --task eta
python run_model.py train --config ../eta-demo/deeptte.json --set runtime.exp_id=train
python tools/make_runtime_fixture.py --directory ../road-demo --task road_representation
python run_model.py train --config ../road-demo/line.json --set runtime.exp_id=train
```

RNN 是轨迹模型族，model.params.rnn_type 可选 RNN/LSTM/GRU；交通 GRU 仍使用 model.name=GRU。轨迹批次提供 lengths，评估在完整 POI 候选集合计算 Recall/MRR/NDCG，padding 不参加排名。默认监控 val_Recall@1、direction=max。

DeepTTE 的输入是已知计划路线几何、距离、用户与出发日历，逐点实际耗时只用于训练目标；不能把实际耗时作为预测输入。数据按完整行程时间区间切分，主损失 MAE/MSE，局部辅助 MAPE 在独立 step 中计算。评估导出 predictions.json，明确 prediction_seconds/truth_seconds。每条路线长度必须达到 kernel_size，当前适配需 .usr 和含 traj_id/coordinates 的 .dyna。

ATSTLSTM 使用具名 atst_ranking 协议：training.loss.name=normalized_bpr，negative_samples 控制训练负样本请求数。训练区间热度驱动无放回采样并排除正类，训练步只计算候选子集；验证/测试对全词表计算确定性 BPR 与排名指标。预测不读取真实下一位置/时刻，默认使用最后已观察位置和查询时刻，模型输入可显式提供已知 query_offset_seconds。BPR loss 与其他轨迹模型的 NLL 不同，比较排名时应使用相同切分与候选集合；细节见 [迁移范围](models_zh.md)。发布示例由 --model ATSTLSTM 生成。

LINE 支持 first/second order。训练使用独立正边集合及负采样，独立评估导出 embeddings.npy 与 node_ids.json；验证指标是持出边损失/准确率，尚不提供下游道路分类基准。STSSL 需要显式设置 output_window=1、input_window>=9，完整续训保留动态损失权重。

三个监督任务均使用相同 train/evaluate/resume 命令。完整基础环境 CLI 验证（五类任务、续训与调参，共 19 条命令）见 [记录](phases34-cli-validation.json)。

SERM 可用 --model SERM 生成小型示例，并复用 train/evaluate/resume 命令。实际数据集的 config.json 需声明 word_vector_file（无表头 GloVe 文本，路径相对数据集目录或使用绝对路径），semantic_column 默认指向 .geo 的 venue_category_name。模型保留并训练预训练向量，按真实 POI ID 提取类别；空/OOV 类别用零语义，前向不构造独热矩阵。该外部文件纳入续训来源版本检查。具体行为与验收见 [模型迁移说明](models_zh.md)。

STAN 可用 --model STAN 生成两级时空注意力示例。stan_ranking 协议采用 training.loss.name=sampled_softmax，negative_samples 请求共享负样本；批内正类先去重，负样本数量按剩余词表截断。验证/测试仍计算完整词表 NLL/排名。查询默认最后观测时刻，可显式提供已知 query_offset_hours，禁止用真实下一事件时间补入；时间上界只取训练前缀。model.params.max_len 必须覆盖最长实际前缀，构建时检查，不静默裁剪。细节见 [模型迁移说明](models_zh.md)。

CARA 可用 --model CARA 生成上下文门控示例。cara_ranking 协议使用逐样本 BPR，默认读取完整已知前缀（history_window=0），可显式限制为最近若干点。时间间隔来自实际时间戳，候选时刻默认最后观测，可在模型输入提供已知 query_offset_hours；实际下一事件不参与预测。训练负采样有界且排除正类，全词表评估不随机抽取候选。门控已按作者实现纠正激活，前缀编码在候选间复用；细节及有意修订见 [模型迁移说明](models_zh.md)。

LSTPM 可用 --model LSTPM 生成历史行程示例，复用 train/evaluate/resume。时间相似度仅用原子训练观测拟合，历史必须在当前行程开始前完成；显式行程边界保留连续历史 LSTM 与两级注意力。空间父索引只选较早观察、同距取最近时间索引，历史质心距离至少 1 km。空历史使用零上下文，首行程样本保留；训练改用排除 padding 类的公共平均 NLL。对旧计算的保留和有意变化见 [模型迁移说明](models_zh.md)。

GeoSAN 可用 --model GeoSAN 生成地理自注意力示例，基础安装即可运行，无需 TorchText/NLTK。data.quadkey_level/quadkey_ngram 定义静态地理词，默认 17/6；geosan_ranking 协议在 training.loss 配置 weighted_binary、negative_samples、num_nearest、temperature 和 exclude_visited。KNN 按正确经纬度弧度计算，候选不足时截断近邻池，访问耗尽有明确回退。训练使用采样候选，验证/测试为完整词表排名及确定性加权二元损失；这与旧最后新位置留出和随机候选协议不同。变长批次、可选候选注意力、训练/独立评估/精确续训均复用公共运行时，详见 [迁移说明](models_zh.md)。

TTPNet 可用 --model TTPNet 生成 ETA 示例，使用相同 train/evaluate/resume 命令。实际数据除计划路线外，还需 .geo 的 JSON embeddings 和 .dyna 的 grid_len、speeds/speeds_relevant1/speeds_relevant2（各 4 值）、speeds_long（7 值）。所有点的 traffic_snapshot_time 必须不晚于该趟出发；可在数据 config.json 用 traffic_snapshot_time_column 指定名称，不能用实际到达时采集的速度冒充出发前信息。元信息 speed_unit 支持 km/h 或 m/s，grid_len_unit/distance_unit 支持 km 或 m。ttpnet_eta 协议用逐路线累计相对 RMSE 训练、整趟 MAPE 验证，指标/JSON 导出按秒。默认 data.min_route_length=3 保持公共 ETA 样本集合，显式设 2 可用两点路线；嵌入行数/宽度由静态文件决定，不固定城市网格大小。详见 [模型说明](models_zh.md)。

Node2Vec/DeepWalk 可用 --model Node2Vec 或 --model DeepWalk 生成示例。训练先安装 `pip install -e '.[embeddings]'`（Gensim>=4.4），使用相同 train 命令；独立 evaluate 指向训练返回的 checkpoints/embeddings.pt，无需 Gensim。road_walk 是全图外部训练协议，不提供验证/调参/完整续训；--resume 与 run_hyper 会在入口拒绝。这里的 training.num_workers 是 Word2Vec worker，确定性模式要求 1；optimizer 为 gensim_sgd，不接受 PyTorch AMP、梯度累积或 TensorBoard 配置。游走在 training/walks.txt 流式保存，优化信息在 training/report.json/metrics.jsonl；输出向量及真实 node_ids，与结构诊断一起保存在 eval。结构诊断不代表预测精度，孤立节点虽保留覆盖但缺少上下文训练信号。详细图权重、目标与能力约束见 [模型说明](models_zh.md)。

## 统一调参

搜索空间为 JSON，例如 `{"training.optimizer.lr":[0.001,0.002],"data.input_window":[4,6]}`。保存为 space.json 后：

```powershell
python run_hyper.py --config ../runtime-demo/gru.json --space space.json --algorithm grid --max-trials 4
```

也可调用 `from libcity.pipeline.runtime_tuning import run_search`，传入 config/space 字典。grid 与带固定种子的 random 都是标准库有限搜索，不需要调参可选包。每个 trial 的配置、数据预处理、权重、训练指标和目录相互独立；trial 只用训练与验证，选定验证最佳权重后测试集只评估一次。search.json 记录所有分数、参数、最佳 trial 和最终测试指标。不支持验证的 HMMM 会在启动前被拒绝；Optuna/Ray 的后端选型仍延后。

## 阶段五交付

完整结果见 [阶段五验收](phase5_acceptance_zh.md) 与 [执行/Commit 映射](phase5_execution_zh.md)。旧扁平配置保存在 reference/config 作为未安装的历史参考，不能传给新入口；旧 registry/Executor/Batch 和检查点迁移命令已删除。当前只接受 best.pt / last.pt，完整状态只能用于续训，纯权重用于独立评估。

独立 CPU 环境在实际只读源码/数据副本上通过全部 148 项测试及仓库外训练/评估/续训。另在 RTX 5060 上验证 GRU、STGCN、DCRNN 的 CUDA 生命周期，含 GRU AMP 续训对照；不据此认证所有模型组合或 MPS。

需要复验时，源码安装 dev/transformer 后运行 `python tools/validate_readonly_runtime.py --directory <新的仓库外临时目录> --report <报告路径>`。它只对生成的副本临时设置权限，实际写入探针必须被拒绝；在会屏蔽 ACL 的沙箱中应使用允许设置临时副本权限的执行环境。它不改动原仓库、原始数据的权限。GPU 验证使用 `tools/validate_cuda_runtime.py`，需要实际可用的 CUDA PyTorch，不会回退 CPU。
