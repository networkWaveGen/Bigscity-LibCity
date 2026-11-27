# 本轮模型迁移范围

本表仅标记阶段一、二的代表组合。未列出的原模型和 37 个特殊交通数据集并未宣称完成迁移。新模型本身为标准 nn.Module，数据适配保留现有窗口/CPT 样本语义。HMMM 是独立直接执行算法。

| 名称 | 任务/数据 | 训练接入 | 检查点/续训能力 | 实际验证级别 |
| --- | --- | --- | --- | --- |
| GRU | traffic_state_pred / point window | 默认 train_loop | 需要权重；完整续训 | 固定旧权重前向/损失、CLI/Python 训练评估续训等价、dropout+shuffle 精确恢复、CPU AMP |
| STGCN | traffic_state_pred / point + 显式静态图 | 默认 train_loop，无模型专用训练函数 | 需要权重；公共完整续训 | 旧权重前向/损失及两次 Adam 更新对照、图 buffer/dtype 迁移、短训及独立评估 |
| DCRNN | traffic_state_pred / point + diffusion buffers | 三参数 curriculum step_fn | 需要权重；完整续训 | 旧计算短训对照、NumPy 课程采样、batches_seen 和下一步参数精确恢复、独立评估 |
| Trafformer | traffic_state_pred / point + 时间特征 | 默认 train_loop | 需要权重；公共完整续训 | full attention 的旧计算与 Adam 更新对照，不等长输入/预测窗、累积尾批更新及独立评估；额外依赖 einops |
| ASTGNN | traffic_state_pred / 现有 ASTGCNDataset CPT | 独立函数顺序调用 teacher/fine_tune 两次 train_loop | 需要权重；两个阶段均可完整续训 | 原尺度标签/损失对照、教师前向与短训对照、TE 梯度、多阶段 LR/重建次数/中断恢复、独立评估 |
| MTGNN | traffic_state_pred / point，子图更新 | 独立 train，复用公共基础工具 | 需要权重；子图/课程进度完整续训 | 全图旧计算与 Adam 更新对照，子图参考参数/LR，OneCycle+课程恢复，静态图小子图、独立评估及导出开关 |
| HMMM | map_matching / geo+rel+usr+dyna+truth | 直接 evaluate 函数 | 无需权重；不支持续训 | 实际 HMMM 两条轨迹、时间戳有/无、road_id=0 指标、单指标、CLI/Python 等价和能力边界 |

全部实际运行在 Windows CPU。CUDA/MPS 路径保留设备抽象，但未完成真实硬件验收；未对所有架构参数组合和真实城市数据集做精度认证。STGCN/Trafformer 的公共续训依据通用循环与检查点验收，不额外声称已重复每一种模型配置的完整恢复测试。

本轮有意修订的旧训练行为与原因见 [执行记录](execution_zh.md)。旧源文件中的神经网络层仍被复用；旧顶层模型和 Executor 不会被新入口实例化。清理旧文件留在阶段五。
