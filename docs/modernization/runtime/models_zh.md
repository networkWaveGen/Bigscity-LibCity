# 新运行时模型迁移范围

当前共 73 个运行时入口，对应原清单中的 75 个任务—模型条目（轨迹 RNN/LSTM/GRU 共用一个入口）。完整 [100 项 CSV](model-migration.csv) 逐项记录实际能力、依赖、验证与后续动作；其余 25 项仍为 pending，不会自动回退到旧运行时。轨迹预测全部 13 个非模板条目、两个 ETA 模型、三个地图匹配算法和六个道路表征模型均已接入。排除七个延期 Common 变体和两个模板后，本期仍有 16 个交通预测模型待迁移。37 个特殊交通数据集的深度整理仍延后。交通数据统一使用目标时间互斥切分。

| 名称 | 任务/数据 | 训练接入 | 检查点/续训能力 | 实际验证级别 |
| --- | --- | --- | --- | --- |
| STDN | traffic_state_pred / 因果局部 volume/OD flow 与周期偏移 | 默认 train_loop | 需要权重；Adam/metric 调度完整续训 | 四组旧网络完整前向/全部梯度、默认 7/3/3/64/128、独立局部块/四流向/历史向量、真实 ID/空间置换/未来隔离/原始零与 NaN、按需窗口/双原始缓存/文件缺口、所有预测步监督、两种 scaler 完整生命周期、单格/单帧/零额外历史、真实尾批 CPU AMP/spawn 与实际 CLI，见 [说明](stdn_zh.md) |
| GSNet | traffic_state_pred / 完整风险网格与部分多图、risk_cpt | 默认 train_loop + 风险分档平方区域目标及评估 | 需要权重；Adam/metric 调度完整续训 | 四组旧核心模块前向/全部梯度、默认 5/256/[64,64]、矩形/双三图/图节点置换、真实未来时间、所有预测步监督、单格/单帧/零协变量/float64、两种 scaler 的实际训练/评估/精确续训、区域原尺度独立复算、真实尾批 CPU AMP/spawn 与实际 CLI，见 [说明](gsnet_zh.md) |
| DSAN | traffic_state_pred / 连续网格窗口、原始非零掩码 | 默认 train_loop + 公共整组 RMSE | 需要权重；dropout/Adam/metric 调度完整续训 | 四组修正旧子模块的独立样本/区域完整前向及全部梯度、SDPA/全遮蔽、局部方块/位置/物理 ID、所有堆叠参与、动态宽度和目标数、原始掩码/缓存/时间与未来隔离、无标签/批次独立/float64/长预测/CPU AMP/spawn、两种 scaler 精确恢复及实际 CLI，见 [说明](dsan_zh.md) |
| CSTN | traffic_state_pred / 带真实网格位置的单流量 OD | 默认 train_loop | 需要权重；Adam/metric 调度完整续训 | 四组修正旧网络完整前向/所有梯度、逐列 GCC、跨样本独立、两个 OD 轴映射、天气空间广播/仅已知未来推进、缓存无关 rel 隔离/未来观测隔离、有无协变量/单格/float64/CPU AMP、两种 scaler 的训练/独立评估/精确续训与实际 CLI，见 [说明](cstn_zh.md) |
| GEML | traffic_state_pred / 起点终点两空间轴的 OD 流量 | 公共 train_loop + 显式边际辅助 step | 需要权重；Adam/metric 调度完整续训 | 四组修正周期记忆的旧完整预测/辅助头/全部梯度、原始正值图/稳定地理图、三种 scaler 和缺失/零掩码的独立原尺度边际目标/梯度、有效批次更新、无标签/批次独立/float64/CPU AMP/spawn、实际训练/独立评估/两种 scaler 精确续训与发布 CLI；验证使用主 OD MSE，详见 [说明](geml_zh.md) |
| DMSTGCN | traffic_state_pred / 两种观测与实际日历的三张动态图 | 默认 train_loop | 需要权重；dropout/BN/Adam/metric 调度完整续训 | 四组修正旧 Conv2d/形状后的完整预测/全部有效梯度/训练 BN（默认 32/512/40 宽度）、独立图张量收缩/有向幂、动态 skip/LayerNorm/传递宽度、真实 UTC 历史时段/训练区间辅助归一化/缓存/未来隔离、删除死层、单层/零阶/多目标头顺序/batch=1/float64、两类归一化精确续训、单节点/时间尾批 CPU AMP、spawn 与实际 CLI |
| CCRNN | traffic_state_pred / 原始训练 SVD Gaussian 图与逐层耦合 | 公共 train_loop + Python 课程 step + 整组 RMSE | 需要权重；Python 课程/Adam/metric 调度完整续训 | 默认 25 隐藏/50 图维的四组旧预测/teacher/全部有效梯度、独立扩散多项式、经济型 SVD/原 Gaussian/两种归一化/常量及单节点、原始训练区间/缓存/未来隔离/无关 rel、零奇异方向可学习/任意图层数/零 hop/多目标/batch=1/float64、无标签/无 RNG/无图属性、两种图完整训练/独立评估/精确续训、全局 RMSE、真实尾批 CPU AMP 与实际 CLI |
| RGSL | traffic_state_pred / 显式静态图与隐式学习图混合循环网络 | 公共 train_loop + 显式采图训练 step | 需要权重；随机图/Adam/调度完整续训 | 默认双层 64 单元及四组受控采样旧前向/全部梯度对照、作者线性直通梯度、修正高阶静态矩阵递推、零图/空采图/零嵌入/单节点有限梯度、低宽度/多目标/batch=1/float64、图基底及节点参数单次准备、确定性无标签/无 RNG/无跨批影响、每次评估一张新图、完整训练/独立评估/精确续训、真实尾批 CPU AMP 与实际 CLI |
| GTS | traffic_state_pred / 训练区间图特征与扩散 GRU | 公共 train_loop + 独立图训练 step/评估适配 | 需要权重；随机采图/课程学习/BN 完整续训 | 默认宽度及四组受控 Gumbel 旧完整预测/原图 BCE/全部梯度/BN 对照、三组扩散梯度、配对线性分解、原始训练特征/无未来目标/缓存/窗口及 batch 独立、大数组只读与检查点描述、确定图无标签评估/每次评估一次图计算、显式 teacher/图正则截止、完整训练/评估/跨截止轮次精确续训、单节点/单特征步/真实尾批 CPU AMP 与实际 CLI |
| HGCN | traffic_state_pred / 静态路网分区与节点/区域双层网络 | 默认 train_loop | 需要权重；分区/dropout/BN 完整续训 | 修正成员汇总后的八组旧层级网络前向/全部有效梯度/训练 BN、默认 32/256/512 宽度、六种静态图公式、真实谱聚类/私有 RNG/显式分区/零图、零度自适应图回退、全成员梯度、历史/核/图阶数/多目标/batch=1/float64、分区设置复用原始缓存/未来目标隔离、训练/独立评估/dropout/BN/Adam/指数调度/RNG 精确续训、单节点/区域/时间尾批 CPU AMP 与跨进程 CLI |
| DGCN | traffic_state_pred / 显式 CPT 与两层动态图卷积 | 默认 train_loop；具名 dgcn_cpt 协议 | 需要权重；dropout/BN 完整续训 | 默认 60 帧、4/8/64 通道修正旧轴后的前向/全部有效梯度与训练 BN 对照、三种 Chebyshev 阶数/奇数卷积核、粗粒度周期轴、任意激活周期/预测窗/特征/头数、batch=1/float64/单节点、拉普拉斯/消除行归一化奇点、实际默认 12 步数据、缓存/训练统计/标签隔离、完整训练/独立评估/dropout/BN/Adam/指数调度/RNG 精确续训、两类单样本 CPU AMP 尾批与实际 CLI |
| STPGCN | traffic_state_pred / 实际日历与位置感知高斯图关系 | 默认 train_loop | 需要权重；全部关系/卷积/选择层完整续训 | 修正注册/真实跳数后的六种旧完整关系计算与全部梯度、默认 64/3/8 网络、关系因子分解、SciPy 独立最短路、真实星期/时段整数历史、因果 padding/样本独立/无标签、多步/多目标/batch=1/float64、互斥切分/训练统计/缓存/标签隔离、全部层实际更新与 Adam/调度器/RNG 精确续训、单节点尾批 CPU AMP、spawn 与实际 CLI |
| STSGCN | traffic_state_pred / 连续三帧同步图与逐步预测头 | 默认 train_loop | 需要权重；图掩码完整续训 | 16 种旧 individual/sharing、GLU/relu、嵌入组合前向及全部梯度、默认四层网络、三帧有向图 buffer、每次前向重建掩码乘积与三次更新、无原地输入修改/跨样本梯度、多步/多目标/batch=1/float64、原尺度 Huber/阈值/零值掩码、完整训练/独立评估/mask/Adam/调度器/RNG 精确续训、距离倒数/缺边/零距离/缓存、单节点尾批 CPU AMP 与实际 CLI |
| ResLSTM | traffic_state_pred / 双流量周期残差、图拓扑与历史外部流 | 默认 train_loop | 需要权重；完整续训 | 修正周期轴/SAME 池化/batch 轴后的旧模块预测与全部梯度、默认及交换流量列、全部池化尺寸奇偶/负输入、多步/多目标/batch=1/float64、无跨样本预测/梯度、历史外部列与未来隔离、训练统计/缓存、完整训练/独立评估/BN/Adam/多步调度/RNG 精确续训、单样本尾批 BN/CPU AMP、服务窗口停运间隔和实际 CLI |
| DKFN | traffic_state_pred / 连续窗口与图/自身双循环流 | 默认 train_loop | 需要权重；完整续训 | 修正逐样本方差及正系数后的旧递推/预测/全部梯度、12 种图阶数/钳制/输入输出维度对照、作者融合式/系数链式梯度、无跨样本梯度、图 buffer/幂/原尺度掩码、已知未来特征滚动、零方差/单节点/孤立图/系数边界、batch=1/float64、完整训练/独立评估/RMSprop/调度器/RNG 精确续训、CPU AMP 和实际 CLI |
| CONVGCN / MultiSTGCnet | traffic_state_pred / 服务周期窗口、逐帧 CPT 与点图 | 默认 train_loop | 需要权重；完整续训 | 修正节点通道/最近分支后的旧公式与全部梯度、独立第二目标/最后完整窗口、有限边距离分组/最大值/常量/零距离/孤立点、图与 ID 对齐/无原地修改、单分支/多步/多目标/batch=1/float64、完整训练/独立评估/优化器/调度器/RNG 精确续训、距离列缓存与来源保护、CPU AMP 和实际 CLI |
| STResNet / ACFM | traffic_state_pred / 原子网格与逐帧 CPT | 默认 train_loop | 需要权重；完整续训 | 修正空间/特征轴后的旧网络前向与全部梯度、STResNet BN 开关及外部输入、ACFM 五种可运行残差模式/共享权重、多变量/多步/单分支/batch=1/float64、训练/独立评估/BN/dropout/优化器/调度器/RNG 精确续训、.grid/.dyna ID 对齐、未来输入隔离、互斥切分/训练统计、缓存复用与损坏重建、文件/时间边界、.grid 变更拒绝续训、单节点尾批 BN/CPU AMP 与实际 CLI |
| GeomGCN | road_representation / 道路边与自环两分区属性重建 | 默认 train_loop + 公共重建损失/评估 | 需要权重；完整续训 | 修正边分区后的独立稠密公式/嵌入/全部梯度、16 种合并设置、全图入度、无 DGL 训练/独立评估/嵌入导出、dropout/优化器/调度器/RNG 精确续训、无标签/batch=1/float64、空图/单节点/自环去重、2,000 节点稀疏图、CPU AMP 与实际 CLI |
| Node2Vec / DeepWalk | road_representation / 稀疏全图随机游走 | external 独立 Gensim 训练回调 | 独立评估需要安全向量权重；无验证/调参/完整续训 | 原加权有向/无向 Node2Vec 游走、修正起点 0 后的 DeepWalk 游走/重启、转移概率与 alias 分布、局部 RNG/有界缓存、真实 Gensim 向量数值与跨进程确定性、流式语料、ID/图/配置权重校验、无 Gensim 独立评估、无 PyTorch 优化器/记录器、孤立节点/共享 2,000 节点稀疏适配、实际 CLI 与能力拒绝 |
| TTPNet | eta / 出发前交通快照与已知计划路线 | 默认 train_loop + 累计相对 RMSE step | 需要权重；dropout 完整续训 | 旧速度/道路/双向 LSTM/累计预测及全部梯度对照、真实变长打包/批次隔离、预训练道路 ID 对齐、出发前快照校验/标签隔离、训练统计与单位换算、逐路线相对 RMSE/精确拟合有限梯度、跨天秒数/周末、完整训练/独立评估/按秒导出/精确续训、两点路线/任意静态嵌入宽度、batch=1/float64/CPU AMP/零速度/常量、实际 CLI |
| GeoSAN | traj_loc_pred / Quadkey 地理自注意力与候选解码 | 默认 train_loop + 加权二元采样 step | 需要权重；采样/dropout 完整续训 | 四种旧有效位置/解码分支分数与全部梯度、TorchText 词表规则冻结及替代、正确经纬度与弧度 KNN、采样分布/耗尽、静态词表/训练访问隔离、因果/padding/混批/候选分块与子集梯度、非默认融合/用户/时间/奇数宽度、完整训练/独立评估/精确续训、float64/CPU AMP/单位置、阻止旧依赖后实际训练、实际 CLI |
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

GeoSAN 保留缩放位置/区域嵌入、两层区域 Transformer、因果序列 Transformer、可选候选注意力解码及加权二元负采样目标。以静态 POI 的 Mercator Quadkey 滑动 n-gram 构建普通词表，保留原 Field(unk_token=None) 的 PAD=0、频次降序/同频词典序规则；默认层级 17、n-gram 6，不再依赖 TorchText/NLTK。修正原 .geo 经度被当纬度及 BallTree 未用弧度的问题。训练使用标准目标时间切分/完整基准样本，静态词表不按未来访问频率重排；可选已访问过滤仅用训练原子观测。KNN 上限按实际词表截断，显式删除正类、均匀有放回采样，访问耗尽回退到非正类近邻，无无限重试；候选缓存限 128 条，不存全 POI 两两距离。

GeoSAN 的前向返回排除 padding 的完整词表原始分数，训练 step 仅计算正类与所采负类；加权 softmax 保留对分数的梯度，并使用真实均匀采样概率。验证/测试采用完整非正类的确定性加权二元损失与公共排名，不将其当 NLL。单 POI 无负类时只训练正类项。命名 geosan_ranking 协议的 negative_samples/num_nearest/temperature/exclude_visited 均在 training.loss，地理编码参数在 data。非默认用户/时间与乘法融合通过显式源投影修正旧维度不匹配；位置单分支的目标同样只使用位置向量，奇数正弦宽度有效。模型只保留实际使用的层和 buffer，区域位置加法不改写输入；不再有恒等大矩阵及未使用层的参数副本。候选分块复用源编码，默认运行不要求独立外层 train 函数。新基准保留首行程和重复访问样本，区别于旧频次过滤/最后新位置留出/随机候选评估，不宣称复现旧实验精度。

TTPNet 保留三支短期速度 CNN/RNN、长期速度 CNN/RNN、速度与道路 LSTM、两层双向 LSTM、LayerNorm 和累计预测 MLP。归一化、速度/长度到预计秒数的换算均在数据层完成，只拟合训练行程；模型只读标准 Tensor 字典并返回 B×1 的整趟预测，训练辅助方法额外返回累计预测和有效掩码。三个短期分支批量处理，累计求和改为一次 cumsum；所有路线 LSTM 使用真实长度打包，最终按各自长度取值，避免旧填充污染双向状态。静态 .geo 的 embeddings 按真实 geo_id 对齐并保持可训练，不再固定为 128×128 个格子、32 维。普通 nn.Module 不持有 scaler、logger、device 或任务统计。

TTPNet 使用具名 ttpnet_eta 协议：训练是每条路线有效累计点的相对 RMSE，再按路线平均；验证 loss 是整趟 MAPE，公共指标和导出按秒。输出仍以训练 time_gap 的均值/标准差作仿射尺度，保留旧预测含义，逆变换只在 loss/评估层。向量范数修复精确拟合时旧 sqrt 的 NaN 梯度，eps 默认 1 秒。实际途中累计耗时只作标签；输入为已知计划几何/长度和出发前交通快照，每个点必须有 traffic_snapshot_time（数据 config.json 可用 traffic_snapshot_time_column 指定列名），所有时间必须不晚于该趟出发。静态图嵌入和短/长速度向量由上游提供，本轮没有新增论文的离线张量分解或图嵌入训练器。默认最短路线 3 点，与 DeepTTE 的完整行程集合一致；data.min_route_length=2 显式启用两点路线，比较基准时须保持同一集合。

TTPNet 短期三列 speeds/speeds_relevant1/speeds_relevant2 每点各 4 值，speeds_long 每点 7 值，均为 JSON；grid_len 非负，零速度沿用 0.2 km/h 约定。元信息 speed_unit 可为 km/h（默认）或 m/s，grid_len_unit 和 distance_unit 可为 km（默认）或 m，统一转换后再拟合统计。坐标在双精度中归一化，时间用 total_seconds，默认周末为星期六/日；存在 holiday 时使用明确 0/1 值。两种 ETA 适配共享完整行程读取与互斥时间切分；公共 ETA 逆变换仅提升低精度类型，保留 float64 调用者精度。新流程沿用公共早停配置，可用 patience=0 关闭；不宣称与旧错误填充、日期处理或完整原论文数据流程数值等价。

Node2Vec/DeepWalk 使用完整的 road_walk 任务协议和 external 执行能力，不创建 PyTorch 优化器、GradScaler、早停或 TensorBoard。Gensim 仍负责原 skip-gram 优化；训练安装 embeddings 可选组（Gensim>=4.4,<5，支持 NumPy 2），独立向量评估不导入 Gensim。Node2Vec 默认有向、p=2/q=1、负采样 5；DeepWalk 默认无向、均匀邻居和重启 0，保留 LibCity 原 hs=1 且 negative=5 的组合，显式设负样本 0 可用仅 HS。CPU 单 worker、运行种子和固定 token 初始化映射支持当前环境跨进程重复；deterministic=true 时多 worker 被拒绝，不承诺跨库版本或平台数值相同。

路网按 .geo 原 ID 顺序映射为稀疏边表/偏移，不建完整 N×N 矩阵。默认二值边；Node2Vec 可显式选 weighted，weight_column 或元信息 weight_col 指定正亲和权重。零边删除、负/非有限权重拒绝，重复有向关系最后一条生效，无向化取双向权重最大值，自环由配置明确选择。DeepWalk 只接受二值图。全图训练不虚构 train/val/test 切分，DataBundle 三个 loader 均为 None；孤立节点仍出现在词表，但向量缺少上下文训练信号。每条游走写入实验 training/walks.txt，Word2Vec 按行重放，二阶 alias 缓存限制为 100,000 个邻居槽；采样使用独立随机状态。旧 DeepWalk 的 start=0 错误、默认 RNG 跨调用共享与 Iterable 导入，以及旧 Node2Vec 的 NetworkX 接口/无向标记已修复，旧全局缓存入口不恢复。

外部训练只保存 checkpoints/embeddings.pt（纯 Tensor 状态，不保存 Gensim pickle），含向量与节点 ID、算法配置和稀疏图身份；映射/图/配置不匹配时拒绝加载。导出 eval/embeddings.npy 与 node_ids.json，关闭预测导出会清理旧文件。metrics 只包含节点数、维度、覆盖率、均值范数、孤立点及无出边点等结构诊断，不是分类、链路预测或泛化精度；覆盖率 1 也不表示孤立节点学到了有用表示。训练损失与来源文件版本另记 training/report.json 和 metrics.jsonl。没有验证协议或完整安全续训状态，因此 supports_validation/supports_resume=false，调参和 --resume 在入口拒绝，无 best_validation/best.pt/last.pt；需要重新训练时新建实验。training 中只接受 Word2Vec 的 epochs/num_workers/window_size/batch_words/subsample/dynamic_window、gensim_sgd 和目标配置，不接受无作用的 AMP/梯度累积等参数。

GeomGCN 保留 LibCity 的四层属性自编码器：两个编码层产生节点嵌入，两个解码层重建属性；不声称实现原论文的几何嵌入空间邻域构建。沿用道路非自环边/每节点一个自环的两个分区，分别投影、按完整有向图的入度作源/目标归一化，再合并分区与多头。原实现把边分区错误地替换成节点诱导子图，并遗漏结果写回，不能把其零结果作为等价目标；本轮按[作者分区聚合实现](https://github.com/bingzhewei/geom-gcn/blob/master/utils_layers.py)的边语义独立重写为 PyTorch index_add，不依赖 DGL。

默认 hidden_dim=144、embedding_dim=32、dropout=0.5 对应原类实际读取值（原 JSON 的 dropout 字段没有被该类读取）；默认不归一化，保留非负重建输出。可以显式选择训练节点拟合的 standard，但最终 ReLU 仍保留。四个合并设置支持 cat/mean；embedding_dim 是每头每分区宽度，实际嵌入宽度乘以第二层选择拼接的分区数/头数。旧第二层拼接会导致后续维度不匹配，新层按实际输出宽度构造；最终重建层固定按分区和头取均值以保持原属性宽度，默认及原有有效 mean 路径不变。完整节点图、范数与边作为 buffer，前向不改图、不写文件；嵌入按公共评估层与真实 node_ids 导出。公共重建损失仅提升半精度类型，float64 输入不再被无条件降为 float32。

STResNet/ACFM 使用具名 grid_cpt 协议，默认单步直接预测，用户可显式扩大 output_window；三段按 closeness/period/trend 顺序、各段由近到远读取。closeness 单位为原始时间片，period 为天、trend 为周，两个 interval 参数分别为天/周的倍数；与已有图模型的逐小时 cpt 采样有明确区分。原子 .geo 必须提供完整零起点矩形 row_id/column_id 和唯一 geo_id；优先读取 .grid 的行列坐标，也支持含 entity_id 的 .dyna。文件行顺序和 geo_id 大小不决定空间位置，输入和输出通过 buffer 映射回真实节点顺序，不创建无用途的稠密邻接矩阵。

旧模型直接 view 将特征与空间混排，新实现显式排列时间/通道/空间轴，输出也从卷积通道恢复到 [batch,horizon,node,feature]；数值对照明确以修复轴顺序后的旧数学模块为参考，不宣称与旧错误布局等价。STResNet 的预测通道不再固定为 2，未启用 BN 时不创建无用的 BN 参数。ACFM 保留共享逐时间残差权重、逆序 ConvLSTM 递推、两级注意力与时刻门控，注意力维度按真实隐藏宽度构建，多步回归和任意目标维度显式输出；没有 local 或 global 分支时只构建实际需要的网络。split-chans 要求隐藏宽度可被观测特征数整除；cpt-sameoutput 保留旧每分支 2 通道含义，仅在各活动段 count*lstm_channels=2 时允许。旧 merge_mode 选项未被使用，不在新预设中保留。

数据输入分为 X（历史网格观测）、history_features（历史日历/全部已加载外部观测）和 future_features（预测时刻已知日历及显式声明的 known_future_columns）；后两者是全网共享的 [batch,time,feature]，不按节点复制。模型不接收未来真实观测或目标，STResNet 只使用未来已知外部特征，ACFM 的历史外部网络和未来门控可以具有不同维度。外部数值保留原尺度，不沿用旧全时间轴 min-max 处理；目标归一化仅拟合训练区间。历史缺失观测只在输入侧以零填充，原始目标 NaN 和零值留给公共掩码；公共预测损失保留 float64，仅提升半精度。原始数组缓存复用既有机制并独立命名，不缓存窗口；损坏字段/形状自动重建，.grid 文件纳入来源版本与续训保护。1x1 网格真实单样本尾批的 BN 使用已有运行统计而不更新方差，避免复制样本。

CONVGCN 保留两层时间投影图卷积、以节点作为通道的 Conv3d 和全连接预测头。原图卷积输出 [batch,projected_time,node,feature] 直接 reshape 会混排节点/投影轴；新实现先显式排列节点通道，保持原图乘法方向并修复布局。图复制后添加单位自环，注册为 buffer；图卷积偏置始终是 Parameter。`convgcn_windows` 协议默认 input_window=10、output_window=1、服务周期偏移 36/180 个原始时间片，等价于旧 time_lag=11 的历史长度；三个等长窗口按 trend/period/closeness 拼接，各段按时间正序。period_steps/trend_steps 是显式服务周期步数，不自动解释为日历日/周。修复旧数据第二个目标错误取通道 0，保留独立通道及最后一个完整目标窗口；不再要求恰好两个特征。

MultiSTGCnet 使用 `multi_stgc` 协议，`frame_cpt` 输入按 closeness/period/trend，最近部分以原始帧计数，日/周部分按采样间隔推导；各段由近到远，保留旧有效时序方向。默认单步，可显式多步。三距离空间分支保留图乘法、两层 LSTM 和融合；时间分支分别使用 interval/daily/weekly 网络，修复旧最近与日周期都调用 daily 的问题。只构建活动分支；没有最近段时省略空间网络，仍可运行日/周预测。原 hidden_size/num_layers 没有参与计算，不作为新超参数；真实 LSTM 宽度仍是节点数乘输入维度，两层单层 LSTM 顺序连接。

`data.graph.name=distance_bands` 显式使用 .rel 原始距离（distance_column 或数据元信息 weight_col，只有一个属性时可自动识别），不把高斯相似度当距离。未列出的边排除，显式零距离保留；每个原点行的有限边按等距值域分为 near/middle/distant，最大值进入远组、常量行进入近组、无边行全零，不修改输入矩阵。重复关系最后一条生效；symmetric=true 取两个方向已观察距离的较小值，self_loops=true 添加零距离自环。也可显式选择 binary，列出边使用单位距离。距离列选择、实际边数、边界含义进入摘要；原始文件变更拒绝完整续训。两模型复用点数据原始缓存、训练区间归一化和互斥目标切分；间隙校验使用前缀计数。常规/日历输入仍要求完整跨度连续；服务周期窗口的停运处理见 ResLSTM 说明，目标区间互斥不变。

本批数值参考明确修正旧错误轴顺序/距离分组/最近分支，不宣称与旧错误完整路径或论文精度等价。CONVGCN 仍有按节点混合的稠密 Conv3d/全连接，MultiSTGCnet 仍有全局节点特征 LSTM 及稠密距离支持；未改成另一种更轻模型，也未认证大城市规模性能。

DKFN 保留图过滤 LSTM、自身 LSTM、邻居细胞权重及[作者实现](https://github.com/Fanglanc/DKFN/blob/master/DKFN.py)的非对称系数融合式，不将其另改为归一化凸组合。原始图按 threshold（默认 1e-4）二值化，保留列度数左乘归一化与图幂/可选钳制；图和过滤矩阵均为 buffer，构造/前向不修改元信息。默认 hops=3、clamp_adjacency=false，RMSprop 学习率 1e-4。

原 torch.var 对整个 batch 求值，导致同一样本的预测和梯度依赖其他样本；新方差仅在各样本的展平观测/图特征流内计算，非退化维度保留样本方差，单元素维度使用零方差。物理融合系数为 softplus(c)，初值为 1；权重中的 c 是无约束坐标。两方差等比例缩放以避免小分母，半精度方差提升到 float32，保留 float64；两流均无经验方差时回退为隐藏预测均值。系数为正避免负方差系数，所有这些变化单独登记，不能把旧全 batch 方差当数值等价目标。

多步预测与旧流程一样每步重新编码滑动窗口；新预测替换目标通道，未知额外观测沿用最后已知值，只有元信息声明的已知未来通道从 future_features 更新，末步不再构造无用的后续窗口。普通 forward 不读取 y，也不持有设备/scaler/logger；标量与 batch=1 不再被 squeeze 丢维度。完整状态通过公共检查点保存，包括变换前 c 的优化器状态；权重、评估和预测导出均使用公共机制，不恢复旧 .m/.tar 加载。

ResLSTM 保留 LibCity 的两条流量残差 CNN、近期第一流量的规范化图分支、可选历史外部网络、节点轴 LSTM 和线性乘性注意力。`reslstm_windows` 协议默认 input_window=5、单步预测、period_steps=72/trend_steps=360，均为原始服务时间片计数；两个流量由 model.params.flow_indices 选择（默认原始观测列 0/1），要求至少两列真实流量，日历不能充当缺失流量。其他原始观测不会自动当全网外部特征；外部列只来自原始观测之后的历史日历/.ext，不读取未来 external/y。

三个季节窗口明确作为 CNN 的三个通道，修复旧 reshape 把周期和帧混排的问题。LSTM 使用 batch_first=True，序列轴是节点而非其他样本；推理和梯度不再跨 batch 传播。池化按[作者的 SAME 设定](https://github.com/JinleiZhangBJTU/ResNet-LSTM-GCN/blob/master/ResLSTM.py)在尾部补齐，实际展平宽度从 ceil(nodes/2)*ceil(history/2) 推导，支持奇偶节点/窗口、单节点、任意显式预测步与目标数。没有下采样的残差单元不再创建无用投影。默认 base_channels=32、recurrent_hidden=128、recurrent_layers=2；外部循环网络保留 LibCity 的两层 LSTM 串联形态，并不宣称与作者 Keras 外部分支完全相同。图一次性规范化为 buffer，不用 np.matrix、矩阵求逆或每次前向重建。单空间元素尾批的 BN 复用公共运行统计回退，和网格模型共用独立的 runtime_norm 模块。

服务时间片窗口的长期跨度可能包含夜间停运。seasonal_windows（ResLSTM/CONVGCN）允许三个窗口之间的未采样区间有停运间隙，但每个窗口内部，以及近期窗口到完整目标跨度均须连续；原子文件边界仍不能跨越，常规/日历窗口仍检查完整输入跨度。该策略写入 sampling_gap_policy；有间隙的数据在元信息中加入 seasonal_segments_v1，避免旧样本语义无声续训。归一化仍仅使用训练区间，目标集合仍严格互斥，原始缓存继续复用。本批数值对照明确修正轴与池化，不声称复现旧错误布局或原论文精度。

STSGCN 保留 individual（每个时间位置独立图模块）和 sharing（所有三帧窗口共享图模块）两种结构，以及 GLU/relu、时空位置嵌入、可选首层嵌入和每个预测步独立的输出头。每个同步块将历史长度减少 2，input_window 必须大于块数的两倍；同一个块内部的图层宽度必须一致，才可对中心帧逐层取最大值，不同块之间允许改变宽度。默认保留四个 [64,64,64] 块、首层宽度 64 和预测隐藏宽度 128；output_hidden_size 可显式修改。不创建关闭的嵌入参数，不再为维持旧类嵌套增加空包装。

三帧局部图保留每帧内部的空间边、相邻帧同节点双向边与权重为 1 的自环。基础图是独立 buffer；启用 use_mask 时，每次 forward 重新计算 adjacency*mask，不保存带有过期计算图的乘积，不需要 retain_graph=True。位置嵌入使用非原地相加，无首层投影时也不会修改调用方输入。模型仅接收历史 X，不读取 y 或预测时刻的真实观测；输出始终为 [B,H,N,F]。

STSGCN 预设显式选择 data.graph.name=inverse_distance，并保留 symmetric=true。原始 .rel 的正距离 d 转为 1/d，未列边为 0，显式零距离按 LibCity 原规则取 1；无权重的 .rel 保持二值图，不把缺边补成全连接。对称化采用当前公共规则 max(A,A.T)，两方向原始距离不相等时相当于采用较近的一条，不沿用旧读取顺序覆盖规则。负距离拒绝，零距离使用带 where 的除法以避免除零告警；图选择进入缓存身份，不能混用原始距离图缓存。

训练损失显式为 training.loss.name=huber、delta=1，在原尺度上计算；旧模型虽然创建了 SmoothL1Loss(beta=rho)，实际 calculate_loss 却没有调用它，rho 从未改变实际目标。新接口不保留这个无效模型参数；调整阈值应修改 training.loss.delta，且使用 Huber 定义而非除以 delta 的 SmoothL1 定义。公共损失另支持 masked_huber，按 y_raw 排除指定原始零值并按有效目标数聚合；这不改变独立评估的指标掩码规则。支持三轮完整训练/独立评估/精确续训以及实际 CPU AMP，不声称真实城市长训精度或全硬件认证。

STPGCN 默认保留 C=64、L=3、d=8、alpha=4、beta=2、Adam 0.002 与原尺度 masked_mae。模型块的关系推断、图卷积、特征选择及每个关系的六个 GaussianComponent 全部以 ModuleList 注册，均进入优化器、设备/精度迁移与检查点。融合 GLU 的宽度改为 C*(L+1)，不再写死四倍 C；L 必须为正，alpha/beta 可为零，beta 大于历史长度时使用真实零填充，不需要裁剪模型能力。预测窗和训练预算由公共任务默认值或显式用户配置决定，不暗中恢复旧扁平配置的 12 步/200 轮。

专用轻量数据适配复用公共原子缓存、目标时间切分、训练归一化与窗口索引。TrafficWindows 额外保留原始 clock 引用，基础样本内容不变；STPGCN 包装器按相同历史索引生成 history_features=[weekday,time_slot]，使用 int64，星期一为 0。时段按原子 UTC 时间戳距当天零点的秒数整除 time_intervals，steps_per_day=86400/time_intervals，要求采样间隔是一天的正整数约数。起始星期、午夜、文件边界和数据间隙不再由样本计数猜测；日历不重复塞进数值 X，预设也不需要启用 time_in_day/day_in_week。模型只读取 X 和历史日历，不读取 y 或预测期真实日历/观测，标准 model_inputs 已支持该键，无需新训练/评估循环。

静态空间跳数按正边的有向拓扑做有界最短路，默认二值图，图权重的大小不作为跳数距离；自身为 0。范围掩码与真实距离分开保存，不再把跳数原地覆盖成 0/1；不可达或超出 alpha 使用独立的 alpha+1 嵌入索引，嵌入表大小为 alpha+2，避免与自身混淆或出现负索引。模型静态 buffer 为 hop_distance 与 range_mask；前向不改变元信息。关系保留 LibCity 原始空间对及最终转置约定，范围掩码按当前目标行的可达邻居应用，并不自动将有向图对称化。

六个高斯的对数分数可分为空间项和时间项之和。因此 exp(spatial+temporal)=exp(spatial)*exp(temporal)，先对整个历史做一次空间聚合，再对每个历史查询做 beta+1 个因果时段加权。每层不再为每个 batch/时刻构建 [B,beta+1,N,N] 关系张量，只需 [N,N] 空间权重和 [B,T,beta+1] 时间权重；没有把高斯权重改成 softmax，也没有近似剪边。保留 float64，half/bfloat16 的高斯分数提升到 float32；不同求和顺序的浮点差异通过完整旧公式前向/全部梯度对照验证。仍保留稠密空间矩阵与空间相对嵌入，不声称大城市性能或论文精度复现。

数值参考明确先纠正旧普通列表不注册、跳数被覆盖、不可达索引、L 非 3 的融合宽度，以及错误日历输入；不声称复现旧错误训练流程。发布示例启用两层小网络。两种数值特征/多个目标、单节点、零跳数/零时间范围、超长 padding、尾批 CPU AMP 和 spawn 均有实际验收。完整训练、独立评估与包含所有新增参数的精确续训使用公共训练与安全检查点。

DGCN 使用具名 dgcn_cpt 协议，默认 horizon=12、closeness=2、period=1、trend=2，因此实际输入为 60 帧，而非普通连续 12 帧。CPT 原始容器按 trend/period/closeness、近到远存储；模型根据只读布局转为旧 ASTGCNDataset 的 closeness/period/trend、每类内远到近，三个非空类别依次复用输入 BN。公共 CPT 已按每个启用类别的真实秒数精确计算位移，要求采样间隔整除相应物理周期、完整预测窗不超过最短启用位移。已验收默认五分钟数据的完整 60 帧/12 步、小时数据的一步例子、2 小时采样的仅日周期 4 步，以及 7 分钟采样的仅周周期。旧小时整除配置的索引与检查点身份保持一致；修正过不精确位移的配置会因数据元信息不同而拒绝旧续训，详见 [CPT 验证](cpt-cli-validation.json)。

默认保留两个动态图块、c_out=64、K=3、Kt=3、四头注意力与 dropout=0.5。图基底是未做度归一化的 D-A 拉普拉斯缩放，有限非负有向图保持原方向；零图/单节点返回有限的 -I。可学习图残差直接注册为 Parameter；逐行归一化保持旧 epsilon=1e-4 加法规则，只在分母接近抵消时加符号保持的下界，按行除法避免构造对角矩阵乘法。静态图、Chebyshev 单位矩阵与时间掩码均是 buffer，LSTM 状态自动随输入设备/精度创建。

每个块以完整 H 帧为一个粗粒度空间注意力通道组，修复旧 view 将周期位置交错的轴错误；除最后一块外的周期各产生一个粗关系，最后 H 帧产生细关系，经 LSTM 推断动态图，再进行 Chebyshev 卷积、门控和分组时间注意力。M 个历史块时，时间掩码/输出头按前 M-2 个单块与末尾两块推导；默认 M=5 对应 12/12/12/24，末尾两块使用长度与步幅均为 2 的卷积降回 H 步并与其余头相加。M=1 使用单块，避免空切片。Kt 支持正奇数，attention_heads 必须整除启用动态图时的 c_out；可关闭任意 CPT 类别，输出维度由真实目标数决定，不再写死 60、48 或 12。

K=1 是仅有单位项的退化卷积，此时不创建无效的空间注意力、图 LSTM 和图残差参数；K>1 保留完整动态图计算。删除从未参与前向的空间注意力 LayerNorm 参数。时间注意力只去掉确定的单通道轴，不用无条件 squeeze；公共单样本 BN 回退扩展到 1d/2d，零复制尾批不再报错，已有 2d BN 状态键和行为保持一致。

默认训练保留 Adam 0.0005、按 epoch 的指数衰减 gamma=0.92，patience=0 表示不早停。原 calculate_loss 的 masked_mse_torch 未指定零值 null，实际只排除缺失值，因此新 training.loss 明确为 masked_mse、mask_value=null，原始零目标仍参与；独立评估的零值掩码另由公共协议控制。模型不持有归一化器或读取 y；全部预测 Tensor、训练/评估/续训共用标准入口。数值参考明确修正粗周期轴与状态精度后比较旧网络，不声称复现旧错误数据轴、完整旧实验流程或论文精度；保留稠密节点关系和时间矩阵的原规模成本。

HGCN 保留节点、区域两级网络、两组图池化块、三次学习式跨级传递、两级跳连及原输出头。默认 channels=32、skip_channels=256、end_channels=512、adaptive_dim=10、graph_order=2、temporal_kernels=[3,2]、dropout=0.3；输入窗默认 12，预测窗按公共任务协议或显式用户配置设置。每个时间核 k 将历史减少 3*(k-1)，末层时间卷积按实际剩余长度收缩到 1；默认核要求输入至少 10 帧，12 帧仍保持原 3 点末层卷积，较短历史可显式选择较小核。保留 Linear(1,output_dim) 再按预测步投影的原多目标头，不把它改成另一个回归网络。

分区选项在 data.graph：cluster_nodes=0 自动取 min(10,N)，正数要求不超过节点数；cluster_labels 可按 .geo 行顺序显式指定每个节点的非负整数类别（重新编号为首次出现顺序），其实际组数须与非零 cluster_nodes 一致。自动分区对静态图的对称平均亲和度运行 SpectralClustering(assign_labels=discretize)，使用独立固定种子；不读取流量或目标。单组、每节点一组和全零连通图有确定的分区规则，谱聚类若未产生指定数量的非空组则明确报错，可提供显式标签。实际标签与成员矩阵进入只读元信息和续训检查，原始缓存身份排除分区设置，改变分区可复用原始时序缓存。

区域图保留 LibCity 的既有算法：先计算 M.T*A*M 的组间权重和，再用全部和的标准差作 Gaussian 缩放并按 cluster_epsilon（默认 0.1）截断。它仍是对“汇总权重”施加高斯核，非恒定情况下零和对应核值 1；不将其解释为新增的物理区域距离。标准差为零时返回有限零图，避免旧 0/0 或无效尺度。新区域输入为每个组所有成员各加一次，修复旧源码仅把第一个节点加两次的错误；与原版对照明确应用了这项修正。

adjtype 支持 doubletransition、transition、symnadj、normlap、scalap、identity，保留旧有向方向/转置和 scalap 的 lambda_max=2 定义。节点/区域静态支持、成员矩阵及单位回退图均为 buffer。自适应图保留 ReLU 内积与行归一化；零度行使用本节点单位行，避免 0*inf，未改成 softmax。每次前向重新生成可微图，不缓存计算图；所有成员、图与输入均不被前向原地修改。旧 h/h_cluster 和 gate 内部卷积未参与前向，已删除，不用无效参数充当模型能力。

时间注意力提取到独立公共 runtime_attention，DGCN/HGCN 共享显式维度及单样本 BN 行为；DGCN 状态键和计算保持原样。模型仅读取历史 X，归一化与原尺度 masked_mae 置于公共运行时。默认训练为 Adam 0.001、epoch 指数衰减 gamma=0.92、早停 patience=50。CPU 实测覆盖谱分区跨进程一致、完整训练/独立评估/dropout/BN/优化器/调度器/RNG 精确续训，以及单节点、单区域、单时间步尾批 AMP。保持原层级计算和区域图约定，不声称论文精度、所有实数据或全部硬件认证。

GTS 保留两层时序 CNN、节点嵌入、二分类边学习器及扩散 GRU 编码/解码网络；默认 rnn_units=64、num_rnn_layers=1、max_diffusion_step=3、kernel_size=10、embedding_dim=100。kernel_size 修正旧 kernal_size 拼写，卷积后历史至少剩一帧，因此图特征长度至少为 2*kernel_size-1。所有输入/输出宽度从元信息推导，不再把多目标输出默认为一维。旧 filter_type 参数实际未参与图传播，旧隐藏 dropout 也未启用，均不保留为无效开关；传播保持对学习图加自环、行归一化后转置的单个随机游走支持及原 Chebyshev 特征顺序。

图特征来自未滑窗的原始训练时间区间，而非旧代码每个训练窗口的第一帧，因此不再受 input_window、output_window、padding 和 batch_size 改变影响。data.graph.history_channel 默认选第 0 个历史观测列，只允许 observed_feature_indices 中的列；目标列使用训练集归一化器，其余原始观测列保持原值，日历和已知未来通道不可用作图学习序列。history_steps=0 使用完整训练区间，正数显式限制到最近若干训练时间点。所选时间段必须连续且数值有限，存在停运间隔时需明确选择足够长的连续尾段。通道、起止时间、形状和归一化方式写入摘要；改变图特征选择可复用原始缓存，验证/测试真实观测不进入图学习。

大数组通过 DatasetMetadata.static_data 的只读 NumPy 快照传入，避免把完整时序展开为元信息中的 Python 数值列表；模型将 node_features、graph_prior、单位阵和非对角掩码注册为 buffer。完整检查点的 data_state 只附加这些静态数组的 shape/dtype 描述，数值随模型状态保存，来源版本、配置和输入布局仍参与续训检查；没有 static_data 的既有模型不增加该字段，保持已有检查点元信息一致。节点对线性层在激活前按 sender/receiver 两半精确分解，保持原参数及梯度，去除两张 N²×N 独热配对矩阵和每条边重复的宽矩阵投影；后续仍有稠密 N² 边与扩散计算。完整训练区间越长，CNN 展平后的 fc 参数越多，仍须按数据规模选择 history_steps，不宣称消除了所有规模成本。

forward 只接受历史 X 并返回预测 Tensor，不读取 y、不采样图。独立评估与验证对二分类边 logits 取 argmax，再去掉自边；每次完整评估仅生成一次图，临时适配器不跨训练更新、检查点载入或设备变化缓存图。训练专用 forward_with_aux 保留直通 hard Gumbel 图采样及原 class-0 logits，温度与课程学习均在 training.strategy 显式配置。teacher 目标仅从训练 step 传入，NumPy 每预测步一个抽样、按 batches_seen 的原阈值以及 Torch 采图 RNG 均可完整恢复。半精度采样提升为 float32，随机数端点夹紧，避免双对数与温度计算溢出。

training.strategy 默认 use_curriculum_learning=true、cl_decay_steps=2000、temperature=0.5、graph_regularization_epochs=200、graph_regularization_weight=1。图正则沿用 sigmoid(raw class-0 logit) 对静态先验的独立 BCE，使用 BCEWithLogits 稳定计算，不改为两类 softmax 的概率 BCE；按显式已完成 epoch 数判定是否启用，不再用批次数反推轮次。无有效预测目标但正则启用时仍可做图先验更新。默认预测目标是原尺度 masked_mae、mask_value=null，保留原始零目标；Adam 0.01、epoch 多步衰减 [20,30,40]/gamma=0.1、梯度裁剪 5 与 patience=100。评估从随机图改为确定图、图特征从窗口首帧改为原始训练区间属于明确的语义修订；受控采样的旧计算对照不等于复现旧错误实验流程或论文精度。CPU 生命周期、精确续训与尾批 AMP 已验证，见 [实际 CLI](gts-cli-validation.json)。

RGSL 保留显式静态图和节点嵌入隐式图的双分支混合、两支通道门控、图循环门与卷积预测头。默认 rnn_units=64、num_layers=2、embed_dim=8、cheb_k=2；输入/输出通道由数据元信息推导，历史与预测长度不再依赖原 input_dim/output_dim 默认值。每支注意力仍按节点均值生成逐样本标量门控，原少于 16 通道时的零宽线性层改为至少一个中间单元；默认宽度的参数形状不变。多目标头保持原 channel/node 布局，不把不同目标混在节点轴。

隐式图 logits 保留 ((cosine+1)/2,1-(cosine+1)/2)，不额外取对数或改用相似度 softmax。训练采样规则经 [作者 RGSL 实现](https://github.com/alipay/RGSL/blob/main/model/RGSL.py) 核对，保留 (logits+Gumbel)/temperature 的硬选择及原始分数线性直通梯度；它不是 softmax 代理梯度。training.strategy.temperature 默认 1，正温度不改变给定噪声下的硬边选择，只缩放这条代理梯度；不能将其解释为 softmax 的平滑程度。指数采样保留 Torch RNG，端点使用有限下界；hard+(scores-scores.detach()) 避免大分数抵消硬值。训练 step 显式调用 forward_with_aux，预测损失仍是原尺度 masked_mae、mask_value=0，没有新增图 BCE 或教师标签输入。

forward 始终只返回预测 Tensor，训练模式下直接调用它也不随机采图、不读取 y。验证/独立评估对上述两个 logits 作 argmax，去掉对角边并在每次完整评估开始只准备一次学习图基底；训练每步重新采图，临时评估适配器不跨调用保留结果。模型不再保存带梯度的 adj/tilde 属性。零向量嵌入使用有界归一化，余弦因舍入越界时夹紧；学习图保持原伪拉普拉斯尺度 max(L)-min(L)，不改成归一化图或特征值尺度。空图时用尺度 1，得到有限 -I；静态零图也使用 -I。静态图保持未归一化 D-A 及最大实部特征值缩放，有向边方向不变。

静态和学习图均按矩阵 Chebyshev 递推构建基底，修复旧静态 cheb_k>=3 使用元素乘法的错误；cheb_k=1 只生成单位项，不再误生成第二项。默认 cheb_k=2 的静态公式保持一致。静态基底、单位阵、非对角掩码注册为 buffer，前向不会原地修改输入或元信息。学习图基底每次前向只生成一次，每个循环层的门/更新节点专属权重只准备一次，时间步之间复用可微张量；末层只保留最终隐藏状态，前层才堆叠供下一层读取的时间序列。原无调用的激活/注意力类不复制到新模型。公共未归一化谱缩放提取到 runtime_graph，DGCN 复用原计算，状态键与数值保持不变。

默认训练为 Adam 0.003、最多 100 轮、patience=15，无额外调度器；用户仍可选择公共调度、累积与 AMP。基础原子缓存、互斥时间切分、训练归一化与安全检查点均使用公共实现，无额外依赖。28 项专项覆盖默认宽度/层数及受控采样的旧预测和全部梯度（高阶静态参考显式修正矩阵递推）、作者代理梯度、退化图/零嵌入、图与权重准备次数、完整训练/评估/随机图 Adam 调度 RNG 精确续训，以及三种真实尾批 CPU AMP。评估确定图、高阶静态递推、小宽度门控和零值回退是有意修订，不声称旧错误实验流程或论文精度等价。模型仍使用稠密图和节点专属权重，内存及谱分解成本没有消失；实际命令见 [RGSL CLI](rgsl-cli-validation.json)。

CCRNN 保留逐层耦合图、每个门/候选分支内的连续图卷积、跨图层注意力、扩散 GRU 编码/解码和零 GO 起点。默认 hidden_size=25、n_dim=50、k_hop=3、n_rnn_layers=1、n_gconv_layers=3；输入/目标宽度由元信息决定。n_dim 实际取 min(n_dim,N)，修复小图默认维度不匹配；n_gconv_layers 可设任意正数，对应层间映射均注册为参数，不再固定两次映射/三张图。只含一层图时省去恒为 1 的注意力权重层，k_hop=0 只计算自身项并省去无效的图因子参数。

data.graph.name 固定为 svd_gaussian，svd_rank 默认 25，表示 SVD 保留分量数量，上限为 min(T*F,N)；摘要 actual_rank 也是实际保留数量，而非非零奇异值数。svd_channels=null 时按原子 data_col 选择全部原始观测列，缺少该元信息时使用全部历史观测列；可显式选择唯一非负列索引。日历/已知未来列不允许进入 SVD；未声明 data_col 且启用历史外部观测时，可用 svd_channels 精确选择所需列。原始训练时间点先按 [T,F,N] 展平为 [T*F,N]，经济型 SVD 避免生成不用的完整时间方阵 U；只取真实训练边界之前的原值，与窗口长度、预测长度、padding 和 batch 无关，不使用验证/测试值，也不把归一化后数据悄悄替换进图生成。

节点表示为 diag(s)*Vh 的转置，距离保持原 Euclidean；核保持 exp(-distance/std(distance)^2)，不是距离平方高斯核。原距离尺度为零时定义节点间相似度为 1，再去掉对角；单节点得到零相似图。normalization 默认 randomwalk，按行度归一化，零度行置零；可选 laplacian，使用 I-D^(-1/2)*A*D^(-1/2)，修复旧实现把数组 shape 当作逆度值的问题，孤立行保留单位项。初始化描述、训练边界、所选通道、形状、分量数和常量回退进入只读布局与运行摘要；data.graph.symmetric/self_loops 额外开关不适用于此派生图。

原始数组缓存使用不读取 .rel 的观测加载器，已有格式错误或缺失的 .rel 均不阻塞该模型；SVD 秩/通道/归一化与窗口设置改变可复用原始缓存，再按当前训练区间重建派生图。完整续训仍沿用全数据目录来源版本的保守检查，未缩减既有来源保护。初始化 support 在数据层只读保存，模型内只在构造时分解；CPU torch.linalg.svd 显式使用经济形式并规范符号，之后仅保留可学习左右因子及各层耦合参数，不把初始化矩阵和带梯度的 graph0/1/2 挂在模型上。零奇异方向仅让右因子从零开始，保持初始乘积，同时保留学习该方向的梯度。

图演化保持两端使用同一个映射矩阵与偏置、LeakyReLU 生成支持，不额外做 softmax 归一化。临时图每次前向重新生成，各图卷积对特征做 Chebyshev 递推并一次堆叠，保持原 channel/多项式顺序，不增加稠密图幂缓存。编码器仅为仍有下一层的层堆叠完整时序，解码器按各层隐藏状态更新，模型不持有 scaler/logger/device。原 unused graph_category、实际只支持 1 的 n_supports、未调用 GraphConvMx 及不可运行 small 路径不复制：旧 small 先对矩形特征减节点方形单位阵，后又把二维 Tensor 当多层图列表传递，不是可用的另一种迁移模式。本实现覆盖 [作者使用的耦合 SVD 路径](https://github.com/Essaim/CGCDemandPrediction/blob/main/models/evonn2.py)。

forward 只读取 X，始终返回预测 Tensor；训练模式直接调用它也不读取 y 或抽取课程随机数。训练 step 显式传入归一化目标与逐步 teacher_mask，保留 Python random 每预测步一次的抽样（包含最后一次未使用的抽样），按 state.batches_seen 和 cl_decay_steps 计算逆 sigmoid 阈值；超长训练使用有限零概率回退。training.strategy.use_curriculum_learning 默认 true，cl_decay_steps 默认 300；关闭时不传入目标、不采样。阈值工具提取到公共 curriculum，GTS 保持其原 NumPy 抽样及数值规则。

默认训练为原尺度 masked_rmse、mask_value=null，保留原始零目标；通过 RootMeanSquaredLoss 在有效累积组和整次评估后统一开方，验证/早停不平均逐批 RMSE。可显式选择公共 MAE/MSE/Huber 或非掩码 RMSE。保留 Adam 0.001、metric plateau（factor=0.2、patience=5、threshold=0.001）、梯度裁剪 5、早停 50 与最多 100 轮；预设不改公共切分或预测窗。新增 plateau threshold 的启动校验。47 项专项覆盖四组旧计算/全部有效梯度（含默认 50 节点、25/50 宽度）、原 Gaussian/归一化、仅训练 SVD/缓存、两种归一化完整训练/评估/课程 RNG 精确续训、实际全局 RMSE 和三种单样本尾批 CPU AMP。原始时间边界、拉普拉斯、退化情况及无标签推理是明确修订，不声称旧错误数据流程或论文精度等价；仍有 dense SVD/节点图与全节点注意力成本。具体命令见 [CCRNN CLI](ccrnn-cli-validation.json)。

DMSTGCN 保留主/辅助两支时间卷积、主图/辅助图/辅助到主图三张动态图、门控、图传播、残差与跳连。默认 num_blocks=4、num_layers=2、kernel_size=2、residual_channels=dilation_channels=32、end_channels=512、embedding_dims=40、order=2、dropout=0.3、skip_channels=8。动态图仍由时间、两端节点和三阶核心张量收缩，经 ReLU 与逐行 softmax 得到；图传播保持原有向索引与普通图幂，order=0 只含自身项，不替换为 Chebyshev 递推或额外静态图。

修复接收四维输入却使用 Conv1d 的门/跳连卷积，统一为对应 Conv2d；辅助到主图的投影输入宽度按前一个图卷积的 residual_channels 构造，支持与 dilation_channels 不同。感受野为 1+num_blocks*(kernel_size-1)*(2^num_layers-1)，短历史只在左侧补零；每层有效长度按真实输入/核/膨胀推导，LayerNorm 与跳连输出头不再写死 13 帧或 416 通道。各层跳连先收集、最后按原逆层序一次拼接；多目标输出头显式按 horizon/target/node 排列，保持单目标旧输出含义。

旧网络的最终跳连在最终图传播之前取值，因此最后图卷积、最后辅助卷积和部分末端归一化不参与输出；原 residual_convs 两组从未使用。新模型删除这些无效参数与运算，单个时间块时只保留有效主干，单块或 order=0 时不创建无效动态图嵌入。删除死层不会改变同权重下的有效预测，但会省去原来无用的 dropout 抽样和 BN 更新，不能据此声称整个旧随机训练序列逐步一致。新运行时的连续/中断训练则已完整精确对照。

输入需要两个不同的真实历史观测，data.features.primary_index 默认 0、auxiliary_index 默认 1，索引必须属于 observed_feature_indices，不能把日历或已知未来列当辅助观测。两个索引位于目标通道范围内时使用公共目标归一化器，范围外的选中观测单独按原始训练区间拟合 data.scaler，再只转换历史 X；均不使用验证/测试值或 padding 样本拟合。其状态、训练边界和索引进入只读布局/续训校验。非选中观测、真实未来 y 和 future_features 不参与前向。

时间索引改为第一个实际历史观测的 UTC time_slot，使用独立 int64 history_features，不再按训练/验证样本序号轮转。这个选择保留常规午夜起始、完整连续旧窗口的起点含义，同时修复非午夜起点、切分丢弃窗口或时间间隔造成的错位。采样间隔须整除一天；时段表大小自动为 86400/time_intervals。日历适配提取为公共 calendar_bundle，STPGCN 原导入名称保留，数值和布局不变。data.graph.name=dynamic，只计算学习图，跳过无关 .rel；特征索引/额外归一化和窗口改变可复用原始缓存，完整续训仍保留全目录来源版本保护。

模型为普通 nn.Module，静态结构与动态图参数跟随标准设备/dtype，不持有 scaler/logger/device。forward 总是返回预测 Tensor，eval 模式不采图或消耗 dropout RNG；训练的 dropout/BN 按标准模块模式执行。normalization 可选 batch/layer，单样本单节点/时间的 BN 使用公共有限回退。默认训练是原尺度 masked_mae、mask_value=null（保留零目标），Adam 0.001/weight_decay=0.0001、metric plateau factor=0.3/patience=10/threshold=0.001、裁剪 5、早停 20、最多 200 轮；预设不改公共切分或预测窗。

33 项专项包含四组旧有效网络的修正后前向/全部有效梯度/训练 BN 对照、默认宽度和层数、独立图收缩与幂传播、真实非午夜 UTC 时间、仅训练辅助统计、缺失/坏 .rel 隔离、缓存/未来目标隔离、未选中输入与标签隔离、可变历史/核/通道/多目标及头顺序、单块/零阶、两种归一化的完整训练/独立评估/dropout/BN/Adam/metric 调度/RNG 精确续训、两类真实尾批 CPU AMP 与 Windows spawn。基本安装即可运行；仍保留三阶核心和批量稠密节点图成本，不声称论文精度或全部硬件认证。实际命令见 [DMSTGCN CLI](dmstgcn-cli-validation.json)。

阶段三、四有意调整见 [新增记录](phases34_execution_zh.md)。旧依赖处理步骤及已完成结果见 [登记](optional_dependencies_zh.md)。
