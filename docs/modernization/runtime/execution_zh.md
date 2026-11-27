# 阶段一、二实施与验收记录

本轮按 [实施方案](plan_zh.md) 执行阶段一和阶段二。基线为 `6ddc6ee`，开发分支为 `refactor/runtime-phases-1-2`，交付目标分支为 `refactor/libcity-modernization`。阶段三到五及后续讨论事项不在本轮范围。

## 阶段一

- [x] 从指定基线建立实施分支，并在独立环境安装源码及 dev 依赖。
- [x] 基线完整测试通过 131 项；环境与结果见 `baseline/environment.json`。
- [x] 提取可复用原子数据夹具，固定 GRU 权重、输入、目标和零值掩码；保存 `baseline/gru.json` 的前向与损失对照。
- [x] 分层配置、路径解析、只读元数据与 DataBundle；配置/数据专项检查 38 项通过，GRU 标准模块复用基线权重进行无标签前向对照。
- [x] 默认训练、评估、产物、纯权重和完整续训检查点。
- [x] GRU CLI/Python 等价运行及阶段一验收：156 项完整测试通过，记录见 `phase1-validation.json`。

阶段一同时验证了 step/epoch/metric 三种调度时机、OneCycle 与 WarmupCosine 续训、按有效目标数加权的累积梯度和尾批更新、CPU bfloat16 AMP、可选 TensorBoard、min/max 最佳权重与最近续训状态分离。纯评估构造路径被检查为不调用训练上下文；独立进程导入不加载 Ray/Hyperopt/DGL/TorchText/TensorBoard。旧 CLI 的两项参数测试已随子命令接口更新，其余原有回归保留。

## 阶段二

- [x] STGCN 静态图 buffer 与无样板训练。
- [x] DCRNN step_fn、课程学习及续训进度。
- [x] Trafformer 梯度累积与尾批更新。
- [x] ASTGNN 多阶段训练、标签提取与反归一化。
- [x] MTGNN 独立训练及单 batch 多次更新。
- [x] step/epoch/metric 调度、完整状态恢复、指标全局聚合及预测开关。
- [x] 一个非交通任务的 DataBundle 与无权重/训练能力约束验证：HMMM 地图匹配。

阶段一、二均已在 Windows CPU 通过验收。逐项依据见 [最终验收报告](acceptance_zh.md)，模型范围见 [迁移表](models_zh.md)，完整机器记录见 `acceptance.json`。本轮不做文件哈希校验，提交不添加 co-author。

## 阶段二首批计算对照

STGCN、DCRNN、Trafformer 均以旧模块的同一份权重进行前向、损失和两次 Adam 更新对照，比较全部参数；新 STGCN/DCRNN 的图矩阵经 buffer 保存，模型没有 scaler/logger/device 属性。DCRNN 使用真实 NumPy 采样做课程学习，并验证中断续训后参数、计数及课程进度与连续训练一致。三者还通过统一 Python 实验入口完成训练和独立权重评估。

有意调整的计算语义：

- STGCN 使用 full-horizon 训练，对照时旧模块显式选 full 模式；旧默认 quick 只训练第一个预测步的方式不作为新默认。
- Trafformer 的标量损失不再在模型内部除以累积次数；统一循环按有效目标数归一化累积，并刷新最后不足一组的梯度。其与合并批次的参数对照采用 SGD；同批旧/新短训对照使用 Adam，避免把不同批大小下近零梯度经 Adam 放大的浮点舍入差异误判成循环语义差异。
- Trafformer 解码长度改为 history+horizon，支持不等长窗口；等长条件下与旧模型对照。所有推理只读取 X 和显式 future_features。
- 图预处理作为 data.graph 配置明确指定，默认二值图，STGCN 预设对称化，避免把旧缺边的 inf 距离直接送入图卷积。时间特征开关与 .ext 文件读取分开控制。上述适配不改变目标窗口所属的集合，时间轴修复仍留在阶段三。

首批图模型及配置数据专项验收 19 项通过；ASTGNN、MTGNN 和非交通适配完成后的最终结果见下文及验收报告。

## 定制训练验收

ASTGNN 采用现有 ASTGCNDataset 的 CPT 采样适配；样本实际历史长度、位置索引和配置请求窗口均显式记录，偏离任务默认窗口时写入日志及 runtime_info。没有合并 ASTGNN/ASTGNNCommon。教师强制与自由滚动微调分成两次 train_loop，微调优化器只在阶段切换时重建一次，监控的全局最佳权重跨阶段保留。教师阶段和微调阶段的中断续训均与连续训练参数、计数、学习率相同。

ASTGNN 的有意修正包括：kernel_size=1 时的因果卷积不再因 `:-0` 得到空张量；新位置编码不再 detach 输入投影，确保该投影可训练；位置表上界按真实每小时采样点数计算，支持不同于 12 的预测窗。关闭 TE 时，以相同权重对照旧 ASTGNN 的前向、教师损失和两次 Adam 更新；开启 TE 时独立验证输入投影的有限非零梯度与完整实验。原始尺度损失对照覆盖 standard scaler。

MTGNN 注册独立 train 函数，按子图更新并组合公共优化器、AMP、裁剪、调度、早停和检查点组件。batches_seen 记录输入批次，optimizer_steps 记录真实参数更新，课程学习子步与节点排列另存 task_state。每次子图更新前清零梯度，修正旧执行器同一 batch 内残留上一次梯度的问题；top-k 不超过当前子图节点数。对照覆盖旧模型相同全图条件下的两次 Adam 更新，以及修正梯度清理的子图参考训练、更新次数与学习率；真实子图课程学习和 OneCycle 中断续训逐参数一致。

定制训练完成时完整回归为 173 项；跨任务和最终核对后为 **184 项全部通过**。历史阶段一 156 项记录保留原语义。

## 最终跨任务与完整状态验收

HMMM 使用 `.geo/.rel/.usr/.dyna` 原子文件及真值路线，DataBundle.train/val 为 None，不创建优化器、TensorBoard、metrics.jsonl 或检查点。CLI/Python 的两条轨迹结果一致，RMF=0、AN=1、AL=1；测试覆盖有/无时间戳、仅选 AN、预测开关与不支持的训练/续训请求。复用的评估器修复了 road_id=0 被当成空值过滤、以及只选 AN/AL 时未初始化结果字典的问题。

最终核对补充了带 dropout 和 shuffle 的逐参数精确续训、真实早停及已经停止后的零更新恢复、裁剪在归一化之后执行、数据集元信息改变时拒绝续训，以及 MTGNN 静态图单节点子图的有限值检查。完整 last.pt 使用格式版本 2，额外记录元信息、采样布局及归一化状态；无需新训练轮次的续训也会在新实验目录保留 last.pt。

MTGNN 预定义邻接仅清除已有对角，不再无条件减去单位矩阵，以免缺自环的小子图出现零分母。TensorBoard 从基础依赖移到可选组，dev 组为旧测试保留它；默认新运行时在不安装该包的独立基础环境中验收，记录为 minimal-environment.json。

## Commit 顺序

1. `4e408b4` — test: capture runtime migration baseline and atomic fixture
2. `fb618b5` — feat: add layered runtime configuration and traffic data bundle
3. `fe2e862` — feat: run complete GRU experiments with resumable native training
4. `0f9f7a1` — feat: migrate graph forecasts and curriculum steps to shared training
5. `1939621` — feat: compose staged and subgraph training with complete task state
6. 本次最终交付提交 — feat: validate direct-task runtime and complete phase acceptance

全部为新提交，不改写原有提交。阶段三到五以及方案第 8 节均未作为本轮完成项。
