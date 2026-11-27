# LibCity 现代化架构重构实施计划

本方案面向 LibCity 的全栈现代化重构，不保留向后兼容性，重点解决当前配置杂糅、执行器代码大量重复、模型与数据强耦合、时间切分泄漏及运行产物分散等问题。

实施基线为 `refactor/libcity-modernization` 分支的 `6ddc6ee`。本期以源码目录直接运行 `run_model.py`（及后续 `run_hyper.py`）为主要使用方式，同时把核心实验逻辑放在包内，为后续打包为 wheel 或通过 Docker 分发预留统一入口。脚本参数、扁平配置、旧 `Executor` 接口、自定义 `Batch`、生成缓存和 `.m`/`.tar` 检查点均可直接替换；继续使用现有 `.geo`、`.rel`、`.dyna`、`.ext` 等原子数据文件作为输入。按阶段分批迁移模型，每批明确记录支持范围与验证级别。

## 1. 现状与改造重点

| 现状 | 要解决的问题 |
| --- | --- |
| 扁平全局配置把命令行参数、模型结构、数据预处理、训练参数与数据集元信息混在同一个字典中；相对路径 `./raw_data/`、`./libcity/cache/`、`./libcity/log/` 硬编码在各处。 | 建立分层配置体系，分离数据集元信息、任务基准协议与模型特征适配；数据目录、缓存目录和运行输出目录可独立配置。 [配置解析](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/config/config_parser.py) |
| 注册了 23 个 `Executor`，其中 `TrafficStateExecutor` 有 16 个直接子类，大量复制外层 `train()` 与 `evaluate()` 循环；各任务优化器与调度器配置命名及步进时机不统一。 | 提供可组合的底层训练工具与一个简明通用的 `train_loop`，特殊训练流程直接注册独立 `train` 函数；统一优化器参数组与学习率调度步进规则。 [注册表](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/registry.py) · [交通执行器](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/executor/traffic_state_executor.py) · [轨迹执行器](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/executor/traj_loc_pred_executor.py) |
| 模型内部持有 `self._scaler`、`self.device` 并依赖自定义 `Batch` 对象；部分模型在推理阶段也读取 `batch['y']`；评估与纯推理路径仍会初始化优化器和 TensorBoard writer。 | 模型回归标准 `nn.Module`，推理输入不依赖目标标签；训练、评估和直接运行算法各自只初始化所需组件。 [RNN 模型](../../../libcity/model/traffic_speed_prediction/RNN.py) · [地图匹配执行器](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/executor/map_matching_executor.py) |
| 交通数据先滑窗展开再按样本数量切分，存在训练/验证/测试目标窗口重叠；滑窗缓存体积庞大且配置变更易失效。 | 先通过轻量适配层完成运行时切换，随后单独提交时间轴切分修复（目标窗口互斥、归一化仅用训练集）并分离已知未来特征与目标标签。 [交通数据集](../../../libcity/data/dataset/traffic_state_datatset.py) · [CRANN 数据集](../../../libcity/data/dataset/dataset_subclass/crann_dataset.py) |
| 普通运行、Hyperopt 与 Ray Tune 各自装配流程，且 Hyperopt 每个试验都运行测试集。 | 所有模式统一经由包内实验入口运行；调参阶段只使用训练集和验证集，选定配置后再单独评估测试集（调参后端改造安排在架构稳定后进行）。 [普通运行与 Hyperopt](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/pipeline/pipeline.py) · [Ray Tune](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/pipeline/tuning.py) |

## 2. 模块目录结构

沿用 LibCity 原有的顶层目录命名，只收紧各目录内部代码负责的范围，方便开发者阅读与迁移：

| 目录 | 负责内容 |
| --- | --- |
| `libcity/config/` | 配置解析、任务基准默认值加载、模型预设合并、路径解析与轻量合法性校验。 |
| `libcity/data/` | 原子数据读取、切分、归一化、缓存与 `DataLoader` 构建，向外部提供统一的 `DataBundle` 与只读 `DatasetMetadata`。 |
| `libcity/model/` | 按任务领域组织的模型实现（标准 `nn.Module`）与模型注册信息。 |
| `libcity/executor/` | 逐步替代原有的 23 个 `*_executor.py`。包含可复用的基础训练组件、默认 `train_loop`、通用评估函数，以及少数需要独立实现 `train` 或直接执行的算法入口。 |
| `libcity/evaluator/` | 各任务评估指标计算、跨批次正确聚合与预测结果格式化。 |
| `libcity/pipeline/` | 统一实验入口（`train`、`evaluate`，以及后续 `tune`），负责配置解析、运行时初始化、组件装配与产物输出；CLI 与 Python API 共用这一层。 |
| `libcity/utils/` | 日志、随机种子、检查点读写、文件系统与归一化器等公共工具。 |

根目录的 [run_model.py](../../../run_model.py)（以及后续 [run_hyper.py](../../../run_hyper.py)）仅负责解析命令行参数并调用 `libcity/pipeline/` 中的入口函数，不包含具体实验逻辑。

## 3. 配置体系与统一测评设计

### 3.1 三层数据与实验配置划分

为了既允许不同模型使用各自需要的特征与图处理方式，又保证所有模型在默认设定下可以基于相同数据切分和评估口径进行公平对比，配置分为三个性质明确的层次：

1. **数据集元信息（Dataset Metadata，只读）**：存放在数据目录内的 `config.json`，仅描述原始数据文件的固有属性（如 `geo_file`、`rel_file`、`data_files`、`ext_file`、`data_col`、`weight_col`、`output_dim`、`time_intervals`），与模型和实验无关，由数据加载模块读取并以只读对象传递给下游。
2. **任务基准评测协议（Benchmark Protocol）**：存放在任务级默认配置中，定义统一测评所必需的共享口径，包括时间切分比例/边界（`split`）、预测窗口（`input_window`、`output_window`）以及评估指标与掩码规则（`evaluation`）。模型预设默认不得改写这些字段；如某类多周期模型确需偏离默认协议，运行时在日志和结果摘要中显式提示。
3. **模型输入适配与实验超参数**：由模型预设和用户配置按区块填写，用于控制该模型所需的归一化方式、时间/外部特征开关、邻接矩阵处理、模型结构以及训练超参数。`batch_size` 和 `num_workers` 归入 `training` / `evaluation`，允许模型预设按显存需求覆盖，不影响评测样本集合。

### 3.2 配置区块与合并规则

实验配置划分为 5 个顶层区块：`data`、`model`、`training`、`evaluation`、`runtime`。各组件只接收自己对应的配置区块与显式传入的数据对象。

- **优先级与合并顺序**：`任务基准默认值 < 模型预设 < 用户配置文件 < 命令行覆写`。
- **组件切换替换规则**：对于带类型选择器的子区块（`optimizer`、`scheduler`、`scaler`、`loss`），当高优先级配置改变了 `name` 字段时，整块替换该子区块而不保留低优先级残留参数；仅在 `name` 相同时逐字段合并。
- **路径与资源读取**：数据目录（`data_dir`）、缓存目录（`cache_dir`）和运行输出目录（`output_dir`）可分别指定；用户配置中的相对路径以配置文件所在目录为基准解析；包内默认配置通过 `importlib.resources` 读取。
- **格式与校验**：首期继续使用 **JSON** 格式，不引入新的外部配置框架依赖；在配置解析入口做轻量结构检查（缺失必填项、字段类型错误、非法枚举值和未知键提示），在启动阶段直接给出明确报错。

## 4. 训练与执行架构

以「新开发者阅读核心代码和 1～2 个模型例子即可上手」为原则，训练层采用**底层可复用基础组件 + 极简默认 `train_loop` + 独立 `train` 函数**的两层结构：

### 4.1 底层基础工具组件

所有组件保持实现简短、相互独立，供默认 `train_loop` 和自定义 `train` 函数自由组合使用：

- **优化器与参数分组（`build_optimizer`）**：支持按模块设置差异化学习率、权重衰减或冻结参数（适配常规模型及未来 LLM 投影层/LoRA 微调）。
- **学习率调度器（`build_scheduler`）**：在配置中显式声明步进时机 `interval`：
  - `step`：每次实际优化器参数更新后步进（适用于 `OneCycleLR`、Warmup + Cosine 等）；
  - `epoch`：每个训练轮次结束后步进（适用于 `StepLR`、`MultiStepLR`、`CosineAnnealingLR`、`ExponentialLR`）；
  - `metric`：每次验证后按指定验证指标步进（适用于 `ReduceLROnPlateau`）。
- **检查点管理（`CheckpointManager`）**：统一读写推理权重与完整续训状态，强制使用安全加载（`weights_only=True`）。
- **早停控制（`EarlyStopper`）**：支持指定验证指标名称及优化方向（`min` 或 `max`）。
- **指标记录（`MetricTracker`）**：默认输出控制台日志与结构化 `metrics.jsonl`，TensorBoard 作为可选记录后端且仅在训练阶段初始化。
- **通用评估函数（`evaluate_loop`）**：管理 `eval()` 模式、`torch.no_grad()`、设备搬运、指标聚合与预测产物导出。

### 4.2 默认 `train_loop` 与扩展方式

默认 `train_loop` 是一个顺序清晰、规模精简的原生 PyTorch 训练循环，内置以下通用基础能力：设备放置（支持 CPU / CUDA / MPS）、AMP 混合精度、梯度累积、梯度裁剪、按 `step/epoch/metric` 调度学习率、支持 `min/max` 方向挑选最佳模型（覆盖交通预测按 `val_loss` 最小化与轨迹预测按 `Recall@K` 最大化）、早停以及完整状态续训。

训练状态对象 `TrainState` 显式记录当前 `stage`、`epoch`、`batches_seen`、`optimizer_steps` 与最佳验证指标。不同模型的接入方式分为三类：

| 模型的训练需求 | 接入方式 | 代表模型 |
| --- | --- | --- |
| 标准监督训练循环 | 只编写模型与配置，直接复用默认 `train_loop`。 | GRU、STGCN、GWNET、多数常规交通/轨迹/ETA 模型 |
| 单步损失需要读取训练进度或调用模型辅助损失 | 编写简短的 `step_fn(model, batch, state)` 传入默认 `train_loop`，继续复用外层循环与优化器更新。 | DCRNN、MegaCRN、TESTAM |
| 多阶段训练、单 batch 多次更新、跨轮次算法状态、非标准训练范式或依赖外部 LLM 框架 | 注册独立的 `train(context)` 函数，自由编写训练逻辑并按需组合底层基础组件（多阶段场景如 ASTGNN 可按顺序调用两次 `train_loop`）；无梯度训练的算法直接注册执行/评估函数。 | MTGNN、STSSL、ASTGNN、GeoSAN、MapMatching、Gensim、重型 LLM 定制模型 |

**能力声明与约束**：在组件注册信息中显式标注 `requires_checkpoint`、`supports_resume` 等能力标记。独立 `train` 函数若未实现续训，当用户传入 `--resume` 时由入口直接报错；接入调参时要求训练入口返回标量验证指标。

**分布式训练的轻量预留**：本期不实现多卡分布式代码，但核心代码保持三项习惯以便后续接入 `torchrun` / DDP 或外部加速库：统一使用传入的 `device`；日志、检查点与产物写出集中在少数工具函数中；`DataLoader` 由数据层统一创建。

## 5. 模型接口与数据层过渡策略

### 5.1 模型的输入与输出

- 模型继承标准 `torch.nn.Module`，`__init__` 仅接收模型结构参数与必要的静态元数据（如 `num_nodes`、`adj_mx`），不再持有 `scaler`、`logger` 或硬编码的 `device`。静态图矩阵通过 `register_buffer` 注册，前向中的临时张量基于输入张量所在设备创建。
- `forward(...)` 统一接收标准 `dict[str, Tensor]` 输入并**始终只返回预测张量 `Tensor`**，推理与评估路径不得依赖目标标签 `y`。
- 存在内部辅助损失的模型（如对比损失、记忆库正则损失、门控损失）额外提供训练期前向方法（如 `forward_with_aux(...)`），由对应的 `step_fn` 或独立 `train` 函数调用，避免在推理和评估循环中引入返回类型分支。
- 评估与训练步统一以结构化结果返回：标量指标字典 `metrics: dict[str, float]` 与产物路径字典 `artifacts: dict[str, Path]`。

### 5.2 数据层过渡与分步演进

为避免在第一阶段同时引入过多变量，数据层分步改造：

- **阶段 1–2（运行时切换期）**：通过轻量适配层复用现有 `TrafficStatePointDataset` 等核心数据集，只完成三件事——支持可配置的 `data_dir` / `cache_dir`、输出统一容器 `DataBundle(train, val, test, metadata)`（部分任务允许 `train`/`val` 为 `None`）、把批次转为标准 `dict[str, Tensor]`。这样可在相同样本和计算设置下准确核对新旧运行时的数值一致性。
- **阶段 3（预测输入与切分修复）**：单独提交数据语义改动——将历史观测、预测时刻已知的未来特征（如时间编码与外部协变量）和训练目标明确分开，确保无标签时可正常推理；按目标窗口时间轴切分训练/验证/测试集，保证目标窗口严格互斥，历史输入允许使用切分点之前的已知前序观测，归一化器仅使用训练集区间拟合；注意在评估和零值掩码判断中保留原始尺度目标值，避免归一化再反归一化导致的浮点误差。
- **阶段 4（其余任务覆盖）**：为 `traj_loc_pred`、`eta`、`road_representation`、`map_matching` 分别提供轻量适配函数输出统一 `DataBundle`，并在阶段 2 末尾提前选取 1 个非交通任务做跨任务接口验证。现有 37 个特殊交通 Dataset 子类与底层缓存结构先做标记，后续按需逐步整理。

## 6. 运行入口、实验目录与检查点

### 6.1 源码运行命令

- 训练：`python run_model.py train --config <配置文件>`，追加 `--resume <续训检查点>` 可从中断处继续训练。
- 独立评估：`python run_model.py evaluate --config <配置文件> --checkpoint <权重文件>`。需要权重的模型由入口校验 `--checkpoint` 存在；地图匹配等无需训练权重的算法可直接运行评估。
- 超参数调参：保留 `python run_hyper.py`，在阶段 4 接到同一套实验入口。

### 6.2 实验文件保存位置

缓存目录与运行目录分离：数据预处理缓存统一写入 `<cache_dir>/`；单次实验的全部产物集中写入 `<output_dir>/<exp_id>/`，源码目录与原始数据目录可挂载为只读。

```text
<output_dir>/<exp_id>/
├── resolved_config.json      # 合并解析后的完整实验配置
├── runtime_info.json         # Python/PyTorch 版本、设备、随机种子、切分摘要等环境信息
├── run.log                   # 本次实验完整日志（不再写入全局 ./libcity/log/）
├── metrics.jsonl             # 按 epoch/step 记录的训练与验证指标
├── tensorboard/              # 可选，仅在训练且启用 TensorBoard 时创建
├── checkpoints/
│   ├── best.pt               # 仅含模型权重 state_dict，用于独立评估
│   └── last.pt               # 最近完整续训状态（模型、优化器、调度器、步数/轮次、随机状态、早停及任务状态）
└── eval/
    ├── metrics.json          # 最终评估指标（扁平/结构化数值）
    ├── metrics.csv           # 表格化评估结果
    └── predictions.npz       # 可选预测产物（按任务格式导出，如 ETA 为 predictions.json）
```

检查点保存与恢复要求：使用验证指标挑选最佳权重时，`best.pt` 仅保存权重；`last.pt` 独立保存最近完整 epoch（或 step）的续训状态，跨轮次累积梯度或维护算法状态的模型同步保存相应状态。恢复测试覆盖中断前后的下一次参数更新、学习率、步数计数与最佳模型选择。

## 7. 实施顺序与验收标准

### 阶段 1：建立基线并跑通 GRU 全闭环

- [x] 从 `6ddc6ee` 建立实施分支，在独立环境中运行 `python -m pip install -e '.[dev]'` 与 `python -m pytest -q`，记录平台、依赖版本与基线测试结果（参考基线：macOS CPU、Python 3.13、PyTorch 2.10 通过 131 项测试）。
- [x] 从现有测试提取可复用的小型原子数据夹具，固定随机种子、输入、目标、掩码和预测步，保存基线前向输出与指标作为对照。 [现有交通流水线测试](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/tests/test_modern_core.py)
- [x] 实现分层配置解析与轻量校验、实验入口 `libcity/pipeline/`、基础工具组件、`train_loop`、通用评估与实验目录/检查点输出，并完成 `TrafficStatePointDataset` 的轻量适配。
- [x] 通过 `python run_model.py train/evaluate` 跑通 GRU 在 CPU 上的训练、保存 `best.pt` 与 `last.pt`、独立评估以及 `--resume` 续训；通过 Python 接口执行等价配置并核对结果一致。
- [x] 验证纯权重文件 `best.pt` 可完成独立评估且不初始化优化器和训练记录器；验证配置缺项、组件切换合并规则与检查点缺失报错清晰；验证未安装调参等可选依赖时可正常运行。

### 阶段 2：验证扩展机制与跨任务适配

- [x] 接入 STGCN，验证静态图矩阵通过元数据显式传入并经 `register_buffer` 管理，普通图模型零训练样板代码直接复用 `train_loop`。
- [x] 接入 DCRNN（自定义 `step_fn` 读取 `state.batches_seen`，验证续训后课程学习进度一致）与 Trafformer（验证内置梯度累积及尾批更新规则）。
- [x] 接入 ASTGNN（验证标签提取、反归一化及多阶段顺序调用）与 MTGNN（验证独立 `train` 函数组合底层工具实现单 batch 多次更新）。从同一初始状态和固定数据出发，核对更新次数、学习率与短训后参数，记录有意调整的训练语义。 [DCRNN](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/executor/dcrnn_executor.py) · [ASTGNN](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/executor/astgnn_executor.py) · [Trafformer](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/executor/trafformer_executor.py) · [MTGNN](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3dcae47/libcity/executor/mtgnn_executor.py)
- [x] 验证 `train_loop` 支持 `step`/`epoch`/`metric` 三类学习率调度器步进与续训恢复；用尾批、缺失值掩码与不同预测步验证指标聚合正确性，并统一预测导出开关行为。
- [x] 在扩大迁移前，挑选 1 个非交通任务（如 `traj_loc_pred` 验证变长序列与按 `Recall@K` 最大化选最佳模型，或 `map_matching` 验证无训练权重的直接评估），检验 `DataBundle`、评估返回结构与入口设计未过度偏向交通预测。

### 阶段 3：分离预测输入与修复时间轴切分

- [x] 明确分离历史观测、预测时刻已知的未来特征与训练目标；验证 GRU 等自回归模型在无标签时可正常预测，多变量测试覆盖已知未来特征与不可提前获取的观测量；Teacher Forcing 通过训练步显式传入归一化后的目标。 [GRU 现有输入](../../../libcity/model/traffic_speed_prediction/RNN.py)
- [x] 按目标窗口时间区间确定样本归属训练、验证或测试，确保目标窗口严格互斥，历史输入可使用切分点之前的已知前序观测；归一化统计量仅在训练集拟合；保留原始尺度目标值以保障零值掩码判断精确。对通用交通数据与 CRANN 均增加目标时间互斥检查。 [交通滑窗与切分](../../../libcity/data/dataset/traffic_state_datatset.py) · [CRANN 数据集](../../../libcity/data/dataset/dataset_subclass/crann_dataset.py)
- [x] 将切分语义改动单独提交，保证模型代码回归在相同样本与计算设置下进行；在真实交通数据集（如 METR_LA）上记录修订后的切分边界、各集合窗口数、配置、种子与短训练指标，供后续对照复现。

### 阶段 4：分批迁移其余任务、模型与调参入口

- [x] 以任务清单和模型审计表为基线，建立模型迁移表，逐项记录训练接入方式、支持能力（验证、续训）、依赖要求与实际验证级别（导入、构建、前向、短训练）。优先迁移复用 `train_loop` 的标准模型，再迁移定制训练模型（如 STSSL 跨轮次状态续训、GeoSAN 独立训练）。 [任务清单](../../../libcity/config/task_config.json) · [模型审计](../audit/model_compatibility.csv)
- [x] 5 类任务均至少跑通一个代表组合：交通状态预测、轨迹下一跳预测（变长批次与排名指标）、ETA（时长单位与 JSON 预测导出）、道路表征（节点嵌入向量产物）、地图匹配（直接评估与路网指标）。对依赖陈旧外部库的模型登记替代或重写方案。
- [x] 将 `run_hyper.py` 接入统一实验入口（调参后端待架构稳定后敲定）。每个 trial 使用独立解析的配置、随机种子和运行目录；数据预处理参数变化时按该 trial 配置准备数据；仅基于验证集指标选择最佳配置，最后再在测试集上单独评估。

### 阶段 5：切换默认入口并清理旧实现

- [x] 迁移随仓库发布的脚本、示例配置与测试，删除已被替换的旧入口、旧 `ConfigParser`、旧 `Batch/BatchPAD`、旧 `Executor` 子类与旧 `.m`/`.tar` 加载路径。同步更新使用文档，说明入口、配置、目录与切分变化。
- [x] 在干净 CPU 环境中从源码执行全部回归测试与端到端实验；切换到仓库外目录通过绝对路径运行 `run_model.py`，验证只读源码/数据目录配合独立可写缓存/运行目录可正常完成训练与评估。有 CUDA 或 MPS 环境时补充代表性运行记录。 [现有验证记录](../validation.md)
- [x] 交付可直接运行的训练、续训、独立评估示例，以及完整的模型迁移表、真实数据短跑记录和测试报告。

> 阶段 1 和阶段 2 通过验收后再全面扩大模型迁移，届时应能通过代码与测试验证：新增常规模型只需编写模型本身与预设配置，即可获得完整训练、验证、续训、检查点与评估导出能力。

### 主线收尾状态（持续实施）

阶段四的清单与代表组合已通过验收，但全量模型迁移仍未完成，不能据此将整份方案判定为全部完成。最新逐项状态与证据见 [主线剩余问题执行记录](remaining-work_execution_zh.md) 和 [模型迁移清单](model-migration.csv)。

- [x] 补齐基础交通缓存的保存、复用与失效。
- [x] 补齐 CRANN 的原始目标、统一模型接口及训练/独立评估/精确续训。
- [ ] 完成所有未延期模型的分批迁移及实际验收；模板不是可训练模型，第 8 节 Common 事项保持延期。
- [ ] 全部迁移完成后的最终完整回归、示例与交付审计。

## 8. 标记为后续讨论与延后处理的事项

以下事项已在本轮设计中预留空间，但不在前期阻塞主线重构，留待后续分批推进：

1. **配置文件格式扩展**：首期保留 JSON，后续评估增加 YAML / TOML 支持（按文件后缀自动识别）。
2. **更细粒度的强类型配置 Schema**：首期仅做轻量结构校验，后续可视需要为框架及模型参数补充更严格的 Schema 定义。
3. **数据层深度重构与缓存升级**：将现有 37 个特殊交通 Dataset 子类逐步整理为「基础规范化时序缓存 + 有效样本索引表 + 按需切片 Dataset + 可选重型变换独立缓存」，降低内存占用并提升缓存复用率。
4. **`*Common` 模型变体合并**：现有 7 组 `*Common` 变体（`ASTGCN`、`MSTGCN`、`ASTGNN`、`STResNet`、`ACFM`、`CONVGCN`、`MultiSTGCnet`）本质为多周期 CPT 输入与单窗口输入的模式差异，先做标记暂不动，后续合并为单一模型类并通过预设区分。
5. **超参数调参后端选型**：在主架构稳定后再决定具体后端（如采用 Optuna 替代历史 `hyperopt` 与旧版 Ray Tune API）。
6. **分发与部署形式**：后续根据实际使用场景确定是否发布 wheel（注册控制台命令）及提供标准 Docker 镜像。
7. **多机多卡分布式与重型 LLM 训练框架集成**：当前通过设备抽象与集中式输出预留空间，后续按需对接 `torchrun` / DDP 或外部加速库。
