# 新运行时模型迁移范围

当前共 49 个运行时入口，对应原清单中的 51 个任务—模型条目（轨迹 RNN/LSTM/GRU 共用一个入口）。完整 [100 项 CSV](model-migration.csv) 逐项记录实际能力、依赖、验证与后续动作；其余 49 项仍为 pending，不会自动回退到旧运行时。37 个特殊交通数据集的深度整理仍延后。交通数据统一使用目标时间互斥切分，三个地图匹配算法均为独立直接执行入口。

| 名称 | 任务/数据 | 训练接入 | 检查点/续训能力 | 实际验证级别 |
| --- | --- | --- | --- | --- |
| LSTPM | traj_loc_pred / 历史行程两级注意力与空间膨胀 LSTM | 默认 train_loop + 公共排名损失 | 需要权重；完整续训 | 原有效路径两级注意力/膨胀递推/全部梯度、显式历史行程边界与跨行程状态、训练原子观测时间相似度、因果空间父索引与近邻同距规则、历史质心距离、空历史保留样本、padding/混批/batch=1/float64、完整训练/独立评估/精确续训、CPU AMP、实际 CLI |
| CARA | traj_loc_pred / 上下文注意力与时空门控 | 默认 train_loop + 独立 BPR pair step | 需要权重；随机负采样完整续训 | 作者兼容门控/候选分数/全部梯度、激活纠正及 128 步稳定性、逐样本 BPR、真实小时差与跨午夜查询、球面小数距离、目标输入隔离、有界负采样/耗尽处理、前缀复用/候选子集梯度、完整训练/评估/精确续训、padding/混批/分块/batch=1/float64、CPU AMP、实际 CLI |
| STAN | traj_loc_pred / 两级时空注意力与逐候选间隔 | 默认 train_loop + 共享候选采样 CE step | 需要权重；随机采样完整续训 | 显式修复掩码后的旧两级分数/全部梯度、球面距离与静态直径/毫米级回退、无真实目标输入、训练时间边界、重复/全类别批次、候选子集与全量梯度、训练/评估/精确续训、padding/混批/分块/batch=1/float64、CPU AMP、实际 CLI |
| SERM | traj_loc_pred / 静态 POI 类别与预训练词向量 | 默认 train_loop + 公共排名损失 | 需要权重；完整续训 | 预训练权重原值保留/可训练、旧语义均值/分类/全部梯度、真实 POI ID 对齐、流式词表子集、空/OOV 类别、外部词向量变更拒绝续训、padding/混批/batch=1/float64、完整训练/评估/RMSprop 精确续训、CPU AMP、实际 CLI |
| ATSTLSTM | traj_loc_pred / 个性化时空注意力与候选几何 | 默认 train_loop + 归一化 BPR step | 需要权重；随机负采样完整续训 | 原有效分支分数/BPR/全部梯度、候选子集与全量梯度一致、无目标时刻/位置输入、训练热度隔离、全词表排名、WGS84/Karney 对照、有限缓存/所有权、padding/混批/batch=1/float64、CPU AMP、spawn、实际 CLI |
| HSTLSTM | traj_loc_pred / 观测间隔与 WGS84 距离 | 默认 train_loop + 公共排名损失 | 需要权重；完整续训 | 修正后记忆递推与全部梯度、显式修正旧记忆公式后的分类对照、浮点分桶插值、训练前缀统计隔离、全词表训练/独立评估/精确续训、padding/混批/batch=1/float64/无标签、退化边界、CPU AMP、实际 CLI |
| STRNN | traj_loc_pred / 观测前缀时刻与逐候选几何 | 默认 train_loop + 公共排名损失 | 需要权重；完整续训 | 原转移矩阵逐候选分数/梯度、无目标时刻/位置输入、训练前缀时间边界、批次/分块独立、全词表 NLL、完整训练/评估/精确续训、batch=1/float64/退化边界、CPU AMP、实际 CLI |
| DeepMove | traj_loc_pred / 当前序列与已完成历史行程 | 默认 train_loop + 公共排名损失 | 需要权重；完整续训 | 4 种旧可用非填充分支的条件分数/梯度、严格历史时间、空历史保留基准样本、混批/独立推理等价、dot/general/concat、RNN/GRU/LSTM 生命周期与精确续训、float64/CPU AMP、实际 CLI |
| STG2Seq | traffic_state_pred / 长短期图序列与时刻/通道注意力 | 默认 train_loop + 显式 teacher step | 需要权重；完整续训 | 独立逐轴图卷积/注意力参考及梯度、完整深度训练/独立评估/BatchNorm 精确续训、显式 teacher/未来协变量、多目标/单节点/batch=1/float64、CPU AMP、实际 CLI |
| TimeMixer | traffic_state_pred / 多尺度季节与趋势混合 | 默认 train_loop | 需要权重；完整续训 | 9 种旧可用分支的前向/梯度、avg/max/conv 与两种通道模式、显式局部归一化、时间 FFT/top_k、奇数尺度/偶数核/多目标/batch=1/float64、完整训练/独立评估/精确续训、CPU AMP FFT、实际 CLI |
| STTN | traffic_state_pred / 空间图卷积与时空注意力 | 默认 train_loop | 需要权重；完整续训 | 原首轮图归一化/预测/梯度对照、静态 buffer 与重复评估不漂移、两层 dropout 精确续训、单节点/多目标/batch=1/float64、CPU AMP、实际 CLI |
| STNN / TGCLSTM | traffic_state_pred / 潜因子动力学、图过滤循环网络 | 默认 train_loop | 需要权重；完整续训 | STNN 三种关系模式与独立时间/节点计算、网格插值梯度及原硬查表对照；TGCLSTM 旧递推/预测/目标/梯度、FFR/普通图与 clamp 分支；两模型多目标/batch=1/float64/无标签、训练/独立评估/精确续训、CPU AMP、实际 CLI |
| ToGCN / SimST | traffic_state_pred / 图时序编码解码、邻域时间注意力 | 默认 train_loop；ToGCN 使用显式 teacher step | 需要权重；完整续训 | 旧前向/梯度、显式 teacher 与 Python RNG/dropout 精确恢复、独立邻域/日历参考、因果 mask buffer、不同窗口、多目标/batch=1/float64、65 样本批次、CPU AMP、实际 CLI |
| TESTAM | traffic_state_pred / 三专家与记忆路由 | 默认 train_loop + 专家/门控辅助 step | 需要权重；完整续训 | 旧专家计算、修正的预测切片/路由及辅助损失梯度对照、两种路由模式、原学习率曲线、原尺度掩码、多目标/多观测/batch=1/float64、完整训练/独立评估/dropout 精确续训、CPU AMP、实际 CLI |
| MegaCRN | traffic_state_pred / 记忆银行+显式未来协变量 | 默认 train_loop + 记忆辅助/课程 step | 需要权重；完整续训 | 旧预测/记忆/辅助目标及梯度对照、训练态纯 forward 无标签且不消耗 RNG、课程 NumPy 随机状态与下一轮逐参数恢复、无协变量/batch=1/float64、CPU AMP、实际 CLI |
| ASTGCN / MSTGCN | traffic_state_pred / 显式 CPT 三分支 | 默认 train_loop | 需要权重；完整续训 | 三分支旧前向/梯度、周期顺序、图 buffer、不同过滤器宽度及多目标、batch=1/float64、完整训练/独立评估/精确续训、CPU AMP 与单节点图、实际 CLI |
| ChebConv / GAT | road_representation / 节点属性重建协议 | 默认 train_loop + 节点重建损失/评估 | 需要权重；完整续训 | 旧前向/梯度、互斥节点划分、训练节点归一化、常量与缺失属性、嵌入导出、完整训练/独立评估/精确续训、float64/CPU AMP、2,000 节点稀疏样例与实际 CLI；ChebConv 声明 CPU/CUDA |
| FPMC | traj_loc_pred / 用户ID+变长历史轨迹 | 默认 train_loop + 公共排名损失 | 需要权重；完整续训 | 原四矩阵分解分数与交叉熵梯度对照、正确 NLL、原子用户词表、全词表排名/导出、batch=1/float64、完整训练/独立评估/精确续训 |
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

FPMC 保留四个嵌入矩阵的个性偏好/一阶转移分数，修复旧实现将未归一化分数直接送入 NLLLoss 的错误；标准 forward 返回去除 padding 类后的 log_softmax，对应公共全词表交叉熵训练及排名协议。个性化数据适配显式加入用户词表，用户必须出现在 .usr 中；没有 .usr 时从轨迹用户建立稳定词表。基础 RNN 数据适配保持默认不需要用户特征。

ChebConv/GAT 保留全图节点属性自编码目标，使用具名 `road_reconstruction` 任务协议，原尺度 MAE/RMSE 与协议名写入运行摘要；不将 LINE 的边采样损失当成重建损失。输入是已观察到的完整图与属性，互斥节点集合仅选择监督目标，因此属于传导式属性重建，未宣称是归纳学习或下游分类。数值统计量仅由训练节点拟合，类别词表作为静态属性定义；缺失数值输入用训练均值填补，原始目标 NaN 不参与损失。图以只读边表和稀疏算子保存，前向不写文件，评估统一导出嵌入。ChebConv 的各图切比雪夫递推独立重置，0 阶有效；GAT 按邻域/注意力头归一化防止全图极值导致其他邻域下溢。`auto` 设备选择只考虑模型声明支持的可用设备。

ASTGCN/MSTGCN 按元信息中的 trend/period/closeness 顺序切分输入，映射到各自原有的周/日/小时分支；图谱基注册为 buffer，时间图卷积按时间轴批量计算。保留各分支注意力、图卷积、残差、归一化和矩阵融合计算。MSTGCN 后续块按时间过滤器的实际输出宽度构造；单节点/零拉普拉斯图使用有限的缩放谱基。运行日志记录 data.sampling，摘要同时记录输入样本数和原始时间轴上的 history_span_points；原 Common 变体未合并。

MegaCRN 的普通 forward 在 train/eval 状态均只返回预测张量；记忆三元组/紧凑性损失与显式目标只在训练 step/forward_with_aux 使用，验证以原尺度预测指标选模。解码协变量只取 future_features 中声明为可提前获知的通道，支持没有协变量的情况；历史编码保留原实现仅使用目标通道的设计。临时状态随输入类型/设备创建，图卷积单位阵类型与支持矩阵一致，课程阈值在极大步数下安全归零。

TESTAM 保留时间、动态图、注意力三个专家与记忆门控。旧 STModel 的隐藏残差通道不再作为预测参与路由；多目标按显式专家轴选择，时间通道来自元信息，未来时刻按实际预测窗和 future_features 构造。当前要求连续窗口且 output_window<=input_window，输出取对应未来区间的尾部槽位。普通 forward 始终只返回预测，辅助专家输出只在训练方法中使用；原始目标零值/NaN 不参与专家或门控监督。inverse_sqrt_cosine 调度器保留原预热/周期曲线，interval=step；修改 hidden_size 时应同步显式调整 scheduler.model_dim，避免把优化设置隐藏在模型内部。

ToGCN 保留两层图卷积、逐时 LSTM 编码、历史解码预热与整批 teacher 选择。普通 forward 不读取目标或消耗 teacher RNG；训练通过显式方法传入目标，已知未来通道取 future_features，其他未来观测仅沿用最后已知历史，避免借用未来标签中的额外观测。邻接图注册 buffer，构造不改变全局 anomaly 设置；输出明确为 B×H×N×D，要求 0<=decoder_t<input_window。

STNN 保留潜因子、关系 refine/discover/固定图、动力学 MLP 与时间解码。时间和节点始终使用明确轴，查表范围来自实际窗口/节点数，forward 不再改写 batch。默认 lookup=linear 用双线性插值替代 ceil 整数查表，让 ffunc 能得到梯度并实际更新；这是明确的算法变更，不能把该版本与旧硬查表实现宣称为数值等价。lookup=hard 保留量化选择用于对照，其 ffunc 不可微；旧 12×41 因子选择完全对齐，但两种模式都修正了旧时间/节点混排和解码轴错误。model.params 和 resolved_config 明确记录选择。

TGCLSTM 将 K 阶邻接/FFR 过滤矩阵及细胞过滤器注册 buffer，按输入设备/类型创建状态，滚动预测不读 y。data.graph.reachability_column 默认 FFR_20min；缺少该列会明确报错，可选其他已有列或显式 null 使用普通图模式。轻量适配按 .geo 顺序对齐端点 ID，未列出的边用 0，避免旧 inf 导致 0×inf。原始缓存继续按源文件元信息失效。未来协变量仅取 future_features，其他未来观测沿用最后历史。训练保留原模型的 MSE 与有限零值目标（mask_value=null）；评估掩码仍遵循公共任务协议，用户可显式另设训练掩码。

STTN 保留图卷积、空间/时间注意力和融合门。空间图注册为 buffer，前向始终归一化原图，不把上次归一化的结果覆盖回去；可学习空间嵌入独立复制，避免与原图共享存储。单节点图采用有限的零中心结果；时刻索引随输入设备创建，按列表收集各时刻图输出后一次 stack，避免逐次 cat。position_size=0 按元信息采样间隔推导容量并覆盖输入窗，显式容量不足会报错。普通 forward 返回预测，模型不持有 scaler/logger/device，重复评估不会改变权重或图。

TimeMixer 保留输入特征压缩、多尺度季节/趋势混合和逐尺度预测融合，支持 avg/max/conv、两种 channel_independence 和可选归一化。尺度长度按实际池化/卷积公式构建，支持奇数长度及零降采样层；移动平均采用保长的左右填充。FFT 沿时间轴执行，按样本/隐藏通道选择 top_k 非 DC 频率，明确指定逆变换长度；AMP 时仅 FFT 内部使用 FP32。归一化统计量在本次 forward 中显式传递，不保存为模型临时属性；四维输出按节点轴广播。预测头只返回 output_dim，修复独立通道的 reshape；标量独立通道保留原预测头语义。移除实际未使用的时间/位置嵌入、label_len/freq/embed 配置及未参与计算的 PDM 层。修复分支使用新语义；9 种原来可用的单目标组合另有数值与梯度对照。

STG2Seq 保留六层长期图卷积、三层短期图卷积、时间/通道注意力与滚动解码。原 reshape 会混排 batch、时间、节点和通道，现在全部显式排列；修正后以独立逐轴公式验证，不宣称与旧混排实现数值等价。图谱基只保存一份 buffer；1 阶与单节点/零拉普拉斯图有效。普通 forward 始终自回归，不读取 y，训练通过显式 teacher 方法传目标；查询只取声明的 future_features，额外历史观测不会被冒充未来输入。要求 1<=window<=input_window；支持任意目标维及没有未来协变量的情形。训练保留原 MSE/有限零值目标，评估使用统一任务协议。

DeepMove 保留双循环网络与历史注意力，历史只来自同用户在当前行程开始前已结束的有效行程，按完成时间拼接。各用户历史存为共享 Tensor 银行，样本引用前缀视图，collate 输出独立 Tensor；重叠行程的未来点不进入历史。无历史样本使用零上下文，保留与普通 RNN 相同的目标时间基准样本。历史长度显式掩码，修复填充位置参加 softmax 导致的混批偏差；补齐 concat 注意力并修复 RNN 分支误建 LSTM 解码器，删除未使用的 fc_attn。只对最后当前状态计算注意力，普通 forward 不读目标，输出去除 padding 类后的完整词表 log_softmax。

STRNN 保留时间/空间上下界转移矩阵、全局初始项与用户/位置嵌入，每个候选分别按静态坐标计算距离及上下文，修复旧 predict 用真实目的地距离构造同一向量的问题。默认查询时刻是最后一次观测，不读 target_tim/current_dis；模型输入可显式提供预先已知的非负 query_offset_hours。elapsed_hours 保留亚小时精度，自动时间边界只由训练前缀拟合，距离边界来自静态 POI 坐标包围盒，均可通过模型参数显式覆盖；超界比例夹紧，零跨度回退到有限正边界。距离保留欧氏坐标度单位，坐标中心化后注册 buffer，JSON 解析替代旧 eval。重复历史位置先聚合、候选按块计算，不构造 B×V×L×H 张量。训练改为全词表交叉熵，修复旧批次总分 logistic 损失耦合样本且无负类监督的问题；不宣称与旧错误预测/损失数值等价。

HSTLSTM 保留仓库原有的单层空间时间 LSTM 分类结构，不把该实现宣称为额外复现了论文的层级扩展。门控仍由位置、时间间隔和距离嵌入共同计算，隐藏状态改为 output_gate*tanh(next_cell)，修复旧代码错用 candidate、记忆不能正确影响输出的问题。时间使用秒，距离保留 geopy 的 WGS84 测地线千米；首点保留 0 秒/0.5 千米约定，仅使用当前已观测前缀，不读取下一事件位置或时刻。分桶上界只取训练前缀并保留旧 100 秒/0.5 千米下界和向上取整规则；桶边界保留浮点值，用设备内 searchsorted/插值替代 Python 列表循环与强制 FP32 转换，验证/测试超界值夹紧。静态桶通过 buffer 管理，填充步不更新状态，只保留最终状态而不堆叠全部隐藏/记忆输出；去除 padding 类后使用公共 NLL、排名和检查点。旧错误记忆公式和整数截断桶不是新计算的数值等价基准。

ATSTLSTM 保留位置/用户嵌入、空间时间线性融合、LSTM、注意力与归一化成对排序目标。标准 forward 返回不含 padding 类的全词表原始分数，修复注意力未屏蔽填充导致的混批偏差；不再读取 target_loc/target_dis/target_tim、loc_neg 或真实下一事件。当前序列只包含已知相邻观测的秒间隔和 WGS84 千米距离，首点保留 100 秒/1 千米约定；候选距离来自最后已观察位置到各静态 POI。默认查询间隔为 0，可显式提供预先已知的 query_offset_seconds，不从真实目标时间补入。

具名 atst_ranking 协议要求 training.loss.name=normalized_bpr，negative_samples 配置请求负样本数；实际数目不超过训练中出现位置数减一，至少需要两个训练位置。热度按训练区间原子点计数，不重复统计展开后的前缀，也不看验证/测试频次。训练每批按热度无放回重采样并排除正样本，训练步显式调用候选子集方法；每条样本负对损失求和、batch 取均值以支持公共梯度累积，负采样 RNG 随完整检查点恢复。验证/测试不采样，对全词表计算确定性归一化 BPR loss 与公共排名指标；该 loss 与其他模型的 NLL 不能直接比较。原样本评估和训练前固定且可能包含正类的负样本已明确改为以上协议，不宣称新训练过程与旧缺陷逐次等价。

候选投影先做代数分解，避免 B×V×H 中间张量；训练只计算所选候选分数。组批器最多缓存 16 个起点到全词表的距离行，不构建 V×V 矩阵，输出通过 stack 拷贝，避免调用者污染缓存。WGS84 数据变换用 NumPy 向量化 [Vincenty 逆解](https://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf)，极点或未收敛点对回退到已有 geopy/Karney；不把未收敛近似值当作结果，相关边界依据 [GeographicLib 说明](https://geographiclib.sourceforge.io/html/python/geodesics.html)。随机全球点、微小距离、日期变更线、极点及近对跖点均与 Karney 核对，未新增依赖。

SERM 保留位置/时间嵌入、语义词向量均值、LSTM、位置回归头与用户偏好的加法融合，使用公共全词表 NLL/排名。旧初始化在载入词向量后又对 Linear 统一随机初始化，已修复为预训练语义参数保持原值且可训练；这属于有意修订，数值对照是在相同权重下验证原语义均值、分类和梯度，并不声称旧错误初始化训练结果等价。POI 词 ID 表作为 buffer，EmbeddingBag 的 mean 模式替代 text_size×text_size 单位矩阵及 B×L×text_size 独热中间量；重复词保留计数，未知词不参与均值，空类别为零向量。直接取 LSTM 的最终有效隐藏状态，移除逐样本拼接与 NumPy/CPU 前向转换；输入只需要已观察位置、日历时间、用户和长度，不读取目标或嵌套文本对象。

数据集 config.json 必须提供 word_vector_file；绝对路径直接使用，相对路径以该数据集目录为基准。semantic_column 默认 venue_category_name，可明确指定其他 .geo 类别列，不再硬编码数据集名称；类别按 .geo 的真实 geo_id/运行时词表对齐，不把 geo_id 当 DataFrame 行号。分词保持小写与空白切分，词表由静态 POI 类别和提供的外部预训练词向量确定，不从未来轨迹文本拟合。向量文件使用无表头 GloVe 文本格式，流式只保留所需词，全部找到后可提前停止；维度从文件读取，缺失资源、已用词的维度/有限数值错误启动即报错。运行摘要记录命中词数、维度和空语义 POI 数；外部词向量路径通过已有来源版本机制进入 last.pt，文件变化在参数更新前拒绝续训，不做文件内容哈希。发布夹具仅包含小型合成三维向量，真实实验需指定实际预训练文件。

STAN 保留用户/位置/每周小时嵌入、上下界时空间隔插值、序列自注意力、候选注意力和可学习位置融合。时间与用户 ID 的 0 明确保留作 padding，每周时间为 1–168，真实用户从 1 开始。旧编码器把时间最大值记录到下界字段，并从真实下一事件构造候选时间；新时间上界仅拟合训练前缀，候选查询默认最后已观察时刻，可显式提供预先已知的 query_offset_hours，不读取真实目标位置或到达时刻。时长保留小时小数，超界插值夹紧，零跨度上界回退 1。填充键在 softmax 前掩码，填充查询与候选位置贡献归零，修复混批长度偏差；这些与旧错误掩码/时间上下界属于有意变化，参考对照明确修复掩码并采用正确上界，不宣称完整旧路径等价。

距离保留半径 6371 千米的球面大圆定义，按真实 .geo 词表对齐。静态空间上界通过单位球反向点近邻得到，不保存 POI×POI 矩阵；重复坐标先去重，亚米级点云用固定大小分块回退，避免近邻欧氏距离舍入低估毫米级直径。中心化坐标注册 buffer，前向按需计算历史对和候选块。原间隔嵌入随即对嵌入轴求和，因此先代数合并其插值，去掉 B×L×L/V×E 中间量。model.params.max_len 是可学习位置向量容量，必须覆盖全部实际前缀；构建阶段检查并报错，不静默截断或改变基准样本。

具名 stan_ranking 协议要求 training.loss.name=sampled_softmax，negative_samples 控制共享负样本请求数。训练步去重批内正类并建立标签映射，从其余完整位置词表无放回采样，请求数按可用数量截断；批次覆盖全部位置时直接使用全部正类，不循环重试，也不重复放入相同正类。候选子集方法返回原始分数供交叉熵，训练不计算全词表前向；标准 forward 返回全词表 log_softmax，验证/测试用确定性 NLL 与公共排名，不随机采样。Torch RNG 随公共完整检查点恢复，移除旧模型自增 seed 和 Python 全局重播种。采样、重复正类修复和无目标推理导致的训练差异已显式记录。

CARA 保留上下文注意力与时间/空间门控，但纠正 LibCity 将门控和候选激活颠倒的转写。以[作者官方 notebook](https://github.com/feay1234/CARA/blob/master/CARA.ipynb)为实现对照，门控使用 Keras 1.2 的 hard sigmoid（斜率 0.2、截断到 0–1），候选使用 tanh，嵌入按作者的标准差 0.01 初始化、门控矩阵使用 Glorot uniform；对应激活定义见 [Keras 1.2 源码](https://github.com/keras-team/keras/blob/1.2.2/keras/backend/tensorflow_backend.py)。其余连接，包括作者代码的投影后融合，按官方实现保留，不把它与论文抽象公式或旧错误激活逐值等价。128 步的前向和梯度已验证有限。

历史时间间隔来自实际观测时间戳，单位为小时，修复按 weekday/weekend 小时编码相减导致跨午夜出现负间隔的问题。候选默认最后已知时刻，可显式提供 query_offset_hours；根据已知周内时刻计算跨日、周末和周回绕后的候选日历嵌入，不读取真实下一位置/时间。经纬度按 JSON 与真实 .geo 顺序读取，距离用保留的半径 6373 千米大圆公式，修复弧度、坐标顺序与整数截断；历史与各候选都从相邻已知位置计算。静态点读取和设备内球面距离已抽为共享函数，STAN/HSTLSTM/ATSTLSTM 原有验收保持通过。

history_window 默认 0，表示使用完整已知前缀，移除旧包装器固定取最前 5 点的行为；显式正值只取最近对应数量的点，并重置首点间隔/距离。前缀只编码一次，每个候选再执行一个门控单元，候选分块计算，不对每个候选重放全部历史。标准 forward 只返回全词表原始分数，不写模型临时属性。具名 cara_ranking 协议要求 training.loss.name=bpr；训练每条样本独立计算 1+softplus(负分数−正分数)，保留常数项且不再对整个 batch 先求和。优先从未访问且非正类位置均匀采一个负类；若访问已覆盖词表，退回到任意非正类，避免无限重试。单位置词表没有负对，训练跳过更新、评估 loss 为 0；公共 epoch 调度器同步跳过无更新轮次。

验证/测试固定评估完整词表，loss 是对全部非正类求平均的确定性 BPR，不是 NLL；训练负采样的 Torch RNG 随检查点恢复。该基准与旧固定正类在第 0 位、随机 101 候选以及使用真实到达时刻的评估不同。作者论文的上下文推荐任务允许给定偏好时刻；本运行时按下一位置预测协议默认只用已知信息，不能把当前短跑或全词表指标当成[原论文](https://eprints.gla.ac.uk/160908/7/160908.pdf)精度复现。

SimST 保留因果时间注意力、节点嵌入，以及出/入邻域的指定邻居与均值特征。邻域由配置选择的静态图生成；星期 one-hot 在模型输入适配中转换为原模型的 weekday/7。标准批次始终保持完整节点维，batch 大小不再决定节点采样语义；时间位置/因果 mask 使用实际输入窗，回归头同时支持任意预测窗及多目标。默认使用任务提供的图，不隐式套用旧 Dataset 的 ID 重排与距离预处理；需要距离图时应在 data.graph 明确配置。

LSTPM 保留 LibCity 的当前 LSTM、空间膨胀 LSTM、历史 LSTM 和两级行程注意力。历史按行程完成时间排序，且必须严格早于当前行程开始；显式行程长度在展平历史中保留边界，LSTM 状态跨历史行程连续传递（与原代码及作者实现一致），时间相似度在每条历史行程内部归一化。48 槽 Jaccard 仅由训练区间的原子观测拟合，不用展开后重复前缀或验证/测试数据；额外 padding 行列为零。每条行程只预计算一次因果父索引，各前缀共享视图；球面距离按原 6371 km 半径、千米三位小数取整，最近点同距时选择最近的先前观察。历史距离保留 LibCity 的最后已知 POI 到算术行程质心、至少 1 km 的定义；这与作者仓库的逐点平均距离有区别，不宣称是作者完整精度复现。空历史用零长期上下文，保留首行程基准样本；完整词表排除 padding 类并使用公共平均 NLL，明确区别于旧 summed NLL。移除未实际调用的旧 dropout 配置，不静默启用。时间相似度为 buffer，模型不存设备/归一化器、不读真实下一事件，训练/评估/续训复用公共组件。

本轮有意修订的旧训练行为与原因见 [执行记录](execution_zh.md)。旧源文件中的神经网络层仍被复用；旧数学实现保留为迁移参考，旧 Executor、配置解析器与批对象已在阶段五删除。

阶段三、四有意调整见 [新增记录](phases34_execution_zh.md)。旧依赖处理步骤见 [登记](optional_dependencies_zh.md)；GeoSAN 的 TorchText Field 替换尚未实施，未声明新运行时已支持 GeoSAN。
