# 新运行时模型迁移范围

当前共 27 个运行时入口，对应原清单中的 29 个任务—模型条目（轨迹 RNN/LSTM/GRU 共用一个入口）。完整 [100 项 CSV](model-migration.csv) 逐项记录实际能力、依赖、验证与后续动作；其余 71 项仍为 pending，不会自动回退到旧运行时。37 个特殊交通数据集的深度整理仍延后。交通数据统一使用目标时间互斥切分，三个地图匹配算法均为独立直接执行入口。

| 名称 | 任务/数据 | 训练接入 | 检查点/续训能力 | 实际验证级别 |
| --- | --- | --- | --- | --- |
| STMatching / IVMM | map_matching / 原子路网+GPS+路线真值 | 直接 evaluate，按轨迹隔离状态 | 无需权重；不支持训练/续训 | 与独立单轨迹旧计算一致、多轨迹重复运行、时间戳输出、尾部无候选/全部无候选/单点轨迹、实际 CLI 与能力检查 |
| STID | traffic_state_pred / 元数据明确的日历通道 | 默认 train_loop | 需要权重；完整续训 | 旧前向/梯度、两个日历开关与不同维度、多变量 batch=1/float64、完整训练/独立评估/精确续训、CPU AMP |
| STAEformer | traffic_state_pred / 数值+时刻+星期整数输入 | 默认 train_loop | 需要权重；完整续训 | 正确星期整数布局的旧前向/梯度、两种预测投影、多变量 batch=1/float64、完整训练/独立评估/精确续训、CPU AMP |
| FreTS | traffic_state_pred / frequency MLP | 默认 train_loop | 需要权重；完整续训 | 单变量旧前向/梯度、多变量嵌入与输出布局、batch=1/float64、完整训练/独立评估/精确续训、CPU AMP |
| STNorm | traffic_state_pred / 空间与时间归一化 | 默认 train_loop | 需要权重；统计量完整续训 | 旧前向/梯度、单节点统计量有限、长输入取最后预测、多变量 batch=1/float64、完整训练/独立评估/精确续训、CPU AMP |
| ATDM | traffic_state_pred / spatial agnostic CNN | 默认 train_loop | 需要权重；完整续训 | 旧前向/梯度、多目标回归头、batch=1/float64、完整训练/独立评估/精确续训、CPU AMP |
| AutoEncoder | traffic_state_pred / point window | 默认 train_loop | 需要权重；完整续训 | 旧前向/梯度对照，完整训练/独立评估/精确续训，batch=1 与 float64 |
| AGCRN | traffic_state_pred / adaptive graph | 默认 train_loop | 需要权重；完整续训 | 旧前向/梯度对照，完整训练/独立评估/精确续训，batch=1 与 float64；cheb_order>=2 |
| TGCN | traffic_state_pred / static graph GRU | step_fn，保留原始正则训练目标 | 需要权重；完整续训 | 旧前向与原尺度目标/梯度对照，稀疏图 buffer，完整训练/独立评估/精确续训，batch=1 与 float64 |
| TrafficRNN / TrafficLSTM | traffic_state_pred / autoregressive | 默认 train_loop + 显式 teacher step | 需要权重；完整续训 | 旧前向/梯度对照，标签隔离，dropout/随机 teacher forcing 完整恢复，batch=1 与 float64 |
| Seq2Seq | traffic_state_pred / encoder-decoder | 默认 train_loop + 显式 teacher step | 需要权重；完整续训 | 相同零起点的旧前向/梯度对照，确定性无标签推理，dropout/随机 teacher forcing 精确恢复，batch=1 与 float64 |
| CRANN | traffic_state_pred / crann 双历史 + 显式未来协变量 | 默认 train_loop | 需要权重；完整续训 | 原版 24 步前向及梯度对照、任意预测窗、batch=1、原尺度掩码、完整短训/独立评估/精确续训；temporal_channel 明确选择历史通道，网格覆盖全部节点 |
| GRU | traffic_state_pred / point window | 默认 train_loop | 需要权重；完整续训 | 固定旧权重前向/损失、CLI/Python 训练评估续训等价、dropout+shuffle 精确恢复、CPU AMP |
| STGCN | traffic_state_pred / point + 显式静态图 | 默认 train_loop，无模型专用训练函数 | 需要权重；公共完整续训 | 旧权重前向/损失及两次 Adam 更新对照、图 buffer/dtype 迁移、短训及独立评估 |
| DCRNN | traffic_state_pred / point + diffusion buffers | 三参数 curriculum step_fn | 需要权重；完整续训 | 旧计算短训对照、NumPy 课程采样、batches_seen 和下一步参数精确恢复、独立评估 |
| Trafformer | traffic_state_pred / point + 时间特征 | 默认 train_loop | 需要权重；公共完整续训 | full attention 的旧计算与 Adam 更新对照，不等长输入/预测窗、累积尾批更新及独立评估；额外依赖 einops |
| ASTGNN | traffic_state_pred / CPT 布局与目标时间切分 | 独立函数顺序调用 teacher/fine_tune 两次 train_loop | 需要权重；两个阶段均可完整续训 | 原尺度标签/损失对照、教师前向与短训对照、TE 梯度、多阶段 LR/重建次数/中断恢复、独立评估 |
| MTGNN | traffic_state_pred / point，子图更新 | 独立 train，复用公共基础工具 | 需要权重；子图/课程进度完整续训 | 全图旧计算与 Adam 更新对照，子图参考参数/LR，OneCycle+课程恢复，静态图小子图、独立评估及导出开关 |
| HMMM | map_matching / geo+rel+usr+dyna+truth | 直接 evaluate 函数 | 无需权重；不支持续训 | 实际 HMMM 两条轨迹、时间戳有/无、road_id=0 指标、单指标、CLI/Python 等价和能力边界 |
| FNN | traffic_state_pred / point window | 默认 train_loop | 需要权重；公共完整续训 | 同权重旧计算前向、短训练 |
| GWNET | traffic_state_pred / point + 自适应/静态图 | 默认 train_loop | 需要权重；公共完整续训 | 修正 Conv1d 后同权重参考前向、buffer、短训练；输出最后时刻的固定预测窗 |
| RNN（RNN/LSTM/GRU） | traj_loc_pred / 变长原子轨迹 | 默认 train_loop + 排名损失/评估 | 需要权重；公共完整续训 | 三类 cell 短训、batch=1、整词表排名、独立评估；GRU dropout/shuffle 精确续训；基础安装 CLI |
| DeepTTE | eta / 计划路线与实际耗时目标 | 默认 train_loop + 显式辅助损失 step | 需要权重；完整续训 | 几何/属性/RNN/双分支前向短训、独立评估、精确续训、按秒导出、基础安装 CLI |
| LINE | road_representation / 边与负采样 | 默认 train_loop + 边损失/评估 | 需要权重；完整续训 | 一阶参考前向/损失、二阶短训/精确续训、向量导出、基础安装 CLI |
| STSSL | traffic_state_pred / history>=9，horizon=1 | 独立 train，公共底层更新工具 | 需要权重；跨轮次完整续训 | 辅助损失短训、DWA 历史损失/权重恢复、随机增强/dropout 下一轮逐参数相同、确定性独立推理 |

CPU 覆盖在 Windows 验证；阶段五另在 RTX 5060/CUDA 验证 GRU、STGCN、DCRNN 的训练、独立评估与续训（GRU 含 AMP）。其他模型及 MPS 未完成真实硬件验收；未对所有架构参数组合和真实城市数据集做精度认证。STGCN/Trafformer 的公共续训依据通用循环与检查点验收，不额外声称已重复每一种模型配置的完整恢复测试。

交通旧 RNN/LSTM 的新名称为 `TrafficRNN`/`TrafficLSTM`，避免与轨迹任务的 `RNN` 入口重名。Seq2Seq 解码起点由随机噪声改为固定零向量，保证独立评估确定性；训练标签只能通过 forward_with_teacher 显式传入。TGCN 保留半均方误差与按有效目标数归一化的 L2 正则训练目标，验证/评估报告公共原尺度 MSE 及所选指标。

日历通道现在通过 DatasetMetadata.input_layout.calendar 显式说明位置，额外观测不会挤占日历索引。STAEformer 将星期 one-hot 转为星期整数后再嵌入，steps_per_day 默认由数据采样间隔推导；STID 修复单独启用日历分支时的嵌入维度计算。FreTS 保留原始频域对角权重收缩，按特征求和嵌入修复多变量广播，回归按明确的 horizon/node/feature 布局输出。STNorm 使用公共原尺度损失及 y_raw 掩码，取最后有效时间位置输出完整预测窗，单节点方差不再产生 NaN。ATDM 回归头同时覆盖预测窗与全部目标维度。

STMatching/IVMM 复用现有候选搜索、空间传递和投票计算；原数学源码中的 temporal_analysis 仍为空实现，本轮未把“保留时间戳”宣称为新增了时间速度评分。新入口按轨迹创建算法状态，避免候选集合串到下一条轨迹，修复尾部空候选访问越界；原子路网适配仍要求双端点路段与路线真值。

本轮有意修订的旧训练行为与原因见 [执行记录](execution_zh.md)。旧源文件中的神经网络层仍被复用；旧数学实现保留为迁移参考，旧 Executor、配置解析器与批对象已在阶段五删除。

阶段三、四有意调整见 [新增记录](phases34_execution_zh.md)。旧依赖处理步骤见 [登记](optional_dependencies_zh.md)；GeoSAN 的 TorchText Field 替换尚未实施，未声明新运行时已支持 GeoSAN。
