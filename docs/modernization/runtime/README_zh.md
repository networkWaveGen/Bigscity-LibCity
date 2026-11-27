# 当前源码运行时

实施依据和逐项状态分别见 [方案](plan_zh.md) 与 [验收记录](execution_zh.md)。本轮入口采用五区块 JSON；原来的扁平 CLI 参数已被子命令替换。旧 Executor、ConfigParser、自定义 Batch 和历史加载路径已在阶段五删除；待迁移模型源码仍保留，但不会回退到另一套运行时。

阶段一、二的 184 项记录保留为历史基线。阶段三修复预测输入和目标时间轴，阶段四接入五类任务代表组合及统一调参。当前范围、依赖与验证级别见 [模型迁移表](models_zh.md)，新增实施记录见 [阶段三、四记录](phases34_execution_zh.md) 和 [验收报告](phases34_acceptance_zh.md)。

当前正在继续完成剩余模型迁移与主线缺口，最新提交、修复说明和验收范围见 [主线剩余问题执行记录](remaining-work_execution_zh.md)。历史阶段验收通过不等同于全模型迁移完成；当前支持范围以迁移 CSV 为准。

道路表征的 ChebConv/GAT 使用具名 `road_reconstruction` 协议，任务默认配置与模型预设分开加载；预设仍不能静默改写协议字段。协议名称及相对于默认边采样协议的差异会写入日志和 runtime_info.json。生成示例可用 `tools/make_runtime_fixture.py --directory <目录> --model ChebConv` 或 `--model GAT`；它们重建静态节点属性并导出全节点嵌入，使用全图批次与互斥监督节点集合，不将此指标解释为下游分类精度。

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

每次实验写入 `<output_dir>/<exp_id>/`，含 resolved_config.json、runtime_info.json、run.log、metrics.jsonl、checkpoints/best.pt、checkpoints/last.pt 和 eval/ 下的 JSON/CSV 指标及可选预测文件。`evaluation.save_predictions=false` 关闭预测导出。TensorBoard 仅在训练且 `runtime.tensorboard=true` 时创建；独立评估不创建优化器、训练记录器或 TensorBoard。

基础安装不要求 TensorBoard。启用该后端时安装 `pip install -e '.[tensorboard]'`；dev 组为可选记录器的验收保留该依赖。Trafformer 单独需要 `pip install -e '.[transformer]'`。

`best.pt` 是纯模型 state_dict。`last.pt` 保存最近完整轮次的模型、优化器、调度器、AMP、TrainState、早停、最佳权重、Python/NumPy/Torch 随机状态和训练 DataLoader 随机状态，全部使用 weights_only 安全加载。续训可以延长普通调度器的 epochs；按总更新数定义的 OneCycle/WarmupCosine 必须保持原定 horizon。数据选择、损失、优化器和累积设置不能在续训时悄悄改变。

last.pt 格式版本为 3，还校验只读数据元信息、采样布局、归一化状态与原子来源文件版本。来源在训练上下文建立时记录文件名、大小和纳秒修改时间，不哈希文件内容；数值或图文件发生版本变化时，续训在载入模型/优化器状态前拒绝，避免沿用另一份数据的最佳指标。数据目录内的名字使用相对路径，保留文件元信息后可移动目录；缓存和运行产物不作为原子来源。文件元信息检查是保守判定，单独改变修改时间也会拒绝，并非内容相等证明。旧格式缺少来源记录，需重新训练；best.pt 的独立评估不受此续训格式升级影响。ASTGNN 的两阶段计划同样必须保持一致。旧 `.m/.tar` 和本轮开发中较早的 runtime 格式不自动迁移。换成其他模型时，CLI 的 model.name 选择新预设并丢弃用户 JSON 中前一个模型的结构参数；其余显式实验参数仍按优先级保留。

`training.optimizer.groups` 接受模块前缀、lr、weight_decay 和 freeze。调度器显式设置 interval=step/epoch/metric。梯度累积按真实有效目标数归一化，并在每轮最后一个不足累积组的尾批完成更新；只有实际优化器更新才步进 step 调度器。默认循环支持最小化或最大化指定监控指标、早停、CPU/CUDA/MPS 设备选择和 AMP。CPU AMP 使用 bfloat16，并实际探测 oneDNN 卷积反向能力；后端不支持时使用原生反向内核，退出后恢复设置，前向仍保留 AMP。硬件覆盖以各批次记录为准；阶段五已有代表模型 CUDA 记录，MPS 尚未认证。

## 预测输入与目标时间轴

交通适配现在记录 `split.semantics=target_time_v1`。按原始时间点确定边界，只有目标窗口完整属于该区间的样本才能进入对应集合；丢弃跨边界目标窗口，历史可使用切分点前的已知观测。归一化只拟合原始训练时间区间，验证和测试不 padding。除默认比例外，可在 data.split 中同时提供 train_end/val_end（ISO 时间戳，右侧区间从该时刻开始）。真实 METR-LA 运行记录见 [报告](metr-la-phase3.json)。

批次的 X 只包含历史；y 是归一化后的预测目标，y_raw 是原始目标。损失与评估用 y_raw 判定零值掩码，不经过归一化再还原。future_features 只含日历和明确声明的已知未来协变量。例如 `data.features={"load_external":true,"known_future_columns":["holiday"]}` 允许从 .ext 读取 holiday 的未来值，其余外部观测只保留历史。GRU 的不可提前获取的辅助观测沿用最后历史值，不读取真实未来；Teacher Forcing 只在显式训练步使用 y。

旧展开滑窗缓存不再使用。`data.cache=true` 将未归一化的原始时序、已知未来特征与图缓存到独立 `cache_dir/traffic-raw-v1/`；命中后不再解析原子数据。特征/图配置隔离缓存；源文件路径、大小或修改时间变化时重建，不做源文件内容哈希。每次实验仍按当前配置重新切分、仅用训练区间拟合归一化并按需生成窗口，因此改变窗口、切分、归一化或 batch_size 可复用同一原始缓存。`runtime_info.json` 的 split.cache 记录 hit/miss/disabled 及文件位置；损坏缓存会重建，写入采用临时文件与原子替换。37 个特殊 Dataset 与重型变换缓存的深度整理仍延后。阶段一、二按旧切分生成的 last.pt 不自动续到新语义；数据/采样元信息检查会拒绝不一致状态。

ASTGNN 使用 data.sampling.name=cpt 及只读采样布局，实际历史长度与请求窗口的差异进入运行摘要。data.graph 控制二值/加权/距离图、对称化和自环，模型预设不会悄悄更改统一切分、窗口或指标协议。

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
