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
- [ ] ASTGNN 多阶段训练、标签提取与反归一化。
- [ ] MTGNN 独立训练及单 batch 多次更新。
- [ ] step/epoch/metric 调度、完整状态恢复、指标全局聚合及预测开关。
- [ ] 一个非交通任务的 DataBundle 与无权重/训练能力约束验证。

最终验收前，未打钩项目均不视为完成。本记录随功能提交更新；本轮不做文件哈希校验，提交不添加 co-author。

## 阶段二首批计算对照

STGCN、DCRNN、Trafformer 均以旧模块的同一份权重进行前向、损失和两次 Adam 更新对照，比较全部参数；新 STGCN/DCRNN 的图矩阵经 buffer 保存，模型没有 scaler/logger/device 属性。DCRNN 使用真实 NumPy 采样做课程学习，并验证中断续训后参数、计数及课程进度与连续训练一致。三者还通过统一 Python 实验入口完成训练和独立权重评估。

有意调整的计算语义：

- STGCN 使用 full-horizon 训练，对照时旧模块显式选 full 模式；旧默认 quick 只训练第一个预测步的方式不作为新默认。
- Trafformer 的标量损失不再在模型内部除以累积次数；统一循环按有效目标数归一化累积，并刷新最后不足一组的梯度。其与合并批次的参数对照采用 SGD；同批旧/新短训对照使用 Adam，避免把不同批大小下近零梯度经 Adam 放大的浮点舍入差异误判成循环语义差异。
- Trafformer 解码长度改为 history+horizon，支持不等长窗口；等长条件下与旧模型对照。所有推理只读取 X 和显式 future_features。
- 图预处理作为 data.graph 配置明确指定，默认二值图，STGCN 预设对称化，避免把旧缺边的 inf 距离直接送入图卷积。时间特征开关与 .ext 文件读取分开控制。上述适配不改变目标窗口所属的集合，时间轴修复仍留在阶段三。

首批图模型及配置数据专项验收 19 项通过；最终阶段二完整验收将在 ASTGNN、MTGNN 和非交通适配完成后执行。
