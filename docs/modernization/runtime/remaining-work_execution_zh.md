# 主线剩余问题执行记录

本轮依据根目录 runtime-architecture-plan_zh.md 和代码核查继续实施，基线为 `2a5ad72`，分支为 `refactor/libcity-modernization`。目标包括所有未延期模型的实际迁移、CRANN 完整数据语义与生命周期、基础缓存。方案第 8 节列出的延期事项保持延期；不把迁移清单或代表组合验收当成全库迁移完成。

已发布提交不改写；补充工作按可独立审查的功能提交。提交不添加 co-author，不做源文件内容哈希校验。

| 编号 | Commit | 提交标题 |
| --- | --- | --- |
| commit1 | 670b325 | fix: persist and reuse split-independent atomic traffic caches |
| commit2 | 956c1d5 | feat: migrate CRANN with explicit inputs and resumable shared training |
| commit3 | 23bf934 | feat: migrate six standard forecasts with native training contracts |
| commit4 | 1a8eb2e | fix: support CPU AMP when oneDNN convolution backward is unavailable |
| commit5 | 0c0a20d | feat: migrate calendar spectral and normalization forecasts |
| commit6 | 2a085e1 | feat: migrate direct map matchers with isolated trajectory state |
| commit7 | c626860 | feat: migrate FPMC with personalized inputs and valid ranking loss |
| commit8 | b60ca0e | feat: migrate sparse road autoencoders with explicit reconstruction protocol |
| commit9 | abdd51b | feat: migrate multiscale graph forecasts with explicit CPT branches |
| commit10 | 8a1de42 | feat: migrate memory forecasts with explicit curriculum and auxiliary step |
| commit11 | bf174ae | fix: cover matrix-only models in CPU AMP backward fallback |
| commit12 | 04e2295 | feat: migrate TESTAM experts with explicit routing supervision |
| commit13 | 26f4110 | fix: select native recurrent forward for unsupported CPU AMP backends |
| commit14 | 547218b | feat: migrate graph sequence and nodewise temporal forecasts |
| commit15 | 42ffc31 | feat: migrate latent and graph-filtered recurrent forecasts |
| commit16 | e81011f | feat: migrate STTN with immutable graph transforms |
| commit17 | e2c728e | feat: migrate TimeMixer with explicit scales and temporal decomposition |
| commit18 | 5244e90 | feat: migrate STG2Seq with explicit graph and attention axes |
| commit19 | 81d5572 | fix: reject resume when atomic source versions change |
| commit20 | 20c3437 | feat: migrate DeepMove with causal history and masked attention |
| commit21 | a1734bd | feat: migrate STRNN with explicit candidate geometry |
| commit22 | e3f8655 | feat: migrate HSTLSTM with causal intervals and corrected memory |
| commit23 | c71f33c | feat: migrate ATSTLSTM with causal candidate ranking |
| commit24 | 1307e31 | feat: migrate SERM with intact pretrained semantic embeddings |
| commit25 | 4bf8c9c | feat: migrate STAN with causal attention and bounded sampling |
| commit26 | 33f5f5f | fix: keep epoch schedules unchanged without optimizer updates |
| commit27 | c9b77ca | feat: migrate CARA with author gates and causal pairwise ranking |
| commit28 | 39ed76b | feat: migrate LSTPM with causal session attention and ranking |
| commit29 | 4fc8211 | feat: migrate GeoSAN with native geography and bounded sampling |
| commit30 | cf56fa4 | feat: migrate TTPNet with departure-known traffic and cumulative ETA training |
| commit31 | a87159c | feat: migrate Node2Vec and DeepWalk with explicit external training |
| commit32 | 1ceae72 | feat: migrate GeomGCN with native partitioned reconstruction |
| commit33 | c4cc695 | feat: migrate grid forecasts with causal CPT inputs |
| commit34 | 85c8280 | feat: migrate periodic graph forecasts with explicit distance bands |
| commit35 | e538689 | feat: migrate DKFN with sample-local variance and causal rollout |
| commit36 | 0a3308d | feat: migrate ResLSTM with independent batches and explicit periods |
| commit37 | 05765a3 | feat: migrate STSGCN with fresh graph masks and explicit Huber loss |
| commit38 | 5972104 | feat: migrate STPGCN with registered layers and factored relations |
| commit39 | 3fbe3d7 | feat: migrate DGCN with explicit CPT axes and dynamic graphs |
| commit40 | ed4daee | fix: compute CPT histories from exact active physical periods |
| commit41 | f17311e | feat: migrate HGCN with complete region aggregation and static graph state |
| commit42 | 1020ca6 | feat: migrate GTS with train-only graph features and explicit graph training |
| commit43 | 8df7954 | feat: migrate RGSL with stable mixed graphs and explicit sampling |
| commit44 | 随本节交付的提交 | feat: preserve RMSE reduction across accumulation and evaluation |

## commit1：修复基础交通缓存（670b325）

问题：`data.cache=true` 原先仅创建目录，每次重复读取原子 CSV；基础预处理缓存未生效。

修改：缓存未归一化时序、已知未来协变量、时间戳、文件边界和邻接图。特征/图配置隔离缓存；根据源文件路径、大小和修改时间失效，不哈希源文件内容。切分、窗口、归一化和训练批次由各次实验重新计算。安全 NumPy 加载禁止 pickle；损坏文件重建；临时文件与原子替换防止读到半文件。运行摘要记录实际缓存状态与位置。

验证：`tests/test_runtime_traffic_cache.py`、`tests/test_runtime_timeline.py`、`tests/test_runtime_tuning.py` 共 **21 passed**。缓存测试验证命中不调用原子读取器、trial 参数变化复用原始缓存并重建窗口、数据/元信息/特征/外部协变量/图变化后的失效和隔离、损坏重建及禁用行为。

## commit2：完成 CRANN 数据语义与新运行时迁移

问题：旧路径缺少原始目标通道，仅验证目标互斥，未接入新训练入口；空间注意力固定为 24 步；batch=1 会被无条件 squeeze 破坏维度；网格列维被误当特征维，仅归一化第一列。

修改：新增标准 nn.Module，仅从 X 与显式 future_features 预测，外部未来协变量需通过 known_future_columns 声明；不读取 y，不持有 scaler/logger/device。data.sampling.name=crann 明确指定 temporal_history，空间历史仍由 input_window 决定；共享目标时间切分、训练区间归一化、y_raw 与原始缓存。temporal_channel 显式选择历史通道，dim_x*dim_y 必须覆盖节点且目标为单通道。注意力按实际 output_window 创建，保留 batch 维，隐状态随输入设备与类型初始化；n_layers_tem 实际控制编码与解码层数。旧参考数据路径同时保留 y_raw 并修复整网格归一化。

验证：CRANN 的 4 项测试及迁移表测试 **5 passed**；原版 24 步前向/梯度对照、无标签/batch=1、精确原尺度零掩码、目标时间互斥、真实原子文件短训练、独立评估和两轮连续/中断恢复参数逐项相等。提供 crann.json 与 --model CRANN 夹具生成命令。迁移范围更新为 16/100 条记录，84 条仍 pending，未声明全库完成。

## commit3：迁移六个标准交通模型

新增 AutoEncoder、AGCRN、TGCN、TrafficRNN、TrafficLSTM、Seq2Seq，均为标准 nn.Module，推理只接收输入字典，公共训练/评估/检查点组件承接训练职责。TGCN 稀疏邻接矩阵注册为 buffer，移除硬编码 device/float32；独立 step 保留原始训练正则目标。AGCRN 初始隐状态由输入创建。交通 RNN/LSTM 显式命名以避开轨迹 RNN 入口。

Seq2Seq 旧实现推理仍读取 y 且随机生成解码起点，现使用固定零起点，纯 forward 不读取标签、不消耗推理随机数；teacher forcing 由训练 step 显式传递归一化目标，交通 RNN/LSTM 同样通过显式训练方法实现。

验证：新增 21 项，逐模型对照旧前向及梯度（Seq2Seq 明确使用相同零起点）；TGCN 同时核对原正则目标；六个模型分别完整训练、独立评估及中断续训参数逐项一致；RNN/LSTM/Seq2Seq 覆盖两层 dropout 与随机 teacher forcing 的恢复；所有模型覆盖 batch=1、无标签确定性及 float64。与既有训练引擎合计 **40 passed**。清单更新为 22/100 已迁移，78 条仍 pending。

前三批完成后的完整回归为 **180 passed，0 failed，0 errors，0 skipped**。CRANN 发布示例实际 CLI 训练两轮、独立评估及续训至第三轮均成功；独立评估指标与训练后评估一致，评估/续训命中原始缓存。机器记录见 [remaining-validation.json](remaining-validation.json)。测试临时目录显式设置在工作区；系统沙箱临时路径下旧相对缓存测试曾遇到路径访问失败，同样测试在工作区独立临时目录 26/26 通过后，完整 180 项通过。

## commit4：修复 CPU AMP 卷积反向后端

新问题：当前 CPU/oneDNN 支持 BF16 卷积前向，却在反向报 `DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2`。STID、STNorm、ATDM 的实际 AMP 短训暴露了此公共问题；仅检查 AMP 前向无法发现。

修复：对 CPU AMP 卷积反向做一次实际小探测，隔离并恢复随机状态；仅后端不支持时，在反向阶段使用 PyTorch 原生卷积内核，保留 BF16 前向与 AMP。公共 backward 组件同时供默认循环、MTGNN、STSSL 使用；上下文退出恢复后端设置，不永久关闭 oneDNN，不修改 CUDA 路径。

验证：STGCN 实际 CPU BF16 训练、两轮连续/中断恢复逐参数相同及后端设置恢复；与既有定制训练、STSSL 测试共 12 项通过。下一批五模型的 AMP 测试也全部通过，随模型迁移独立记录。

## commit5：迁移日历、频域与归一化交通模型

新增 STID、STAEformer、FreTS、STNorm、ATDM，全部使用标准 nn.Module 与公共训练闭环，推理不读取标签。新增只读 calendar 通道元信息，按位置读取时刻与星期，避免新增观测列后索引错位。

明确修正的旧实现问题：STID 单独启用时刻/星期分支时嵌入维度交换；STAEformer 把星期 one-hot 的第一列当作星期整数，以及默认日采样数脱离元信息；FreTS 多变量 token 嵌入广播错误与多目标回归的维度重排；STNorm 单节点无偏方差导致 NaN、长输入残留多个时间位置及 n_pred 与预测窗/目标维度脱节；ATDM 回归头未包含全部目标维度。STNorm 的损失统一使用原尺度目标与公共掩码；FreTS 保留原始频域对角收缩，不更换数学算法。

验证：新增 **26 项通过**，包含五模型前向与梯度参考对照（STAEformer 使用正确的星期整数布局）、各自训练/独立评估/精确续训、多变量 batch=1/float64、CPU AMP；另外覆盖 STAEformer 两种投影、STID 独立日历开关和不同嵌入宽度、STNorm 单节点统计量。迁移表为 27/100 已迁移，73 条 pending，仍未完成全库迁移。

日历布局元信息扩充后，续训仍严格校验输入布局；旧布局不一致的 last.pt 会明确拒绝，纯 best.pt 权重评估仍须匹配模型结构。

前五批之后完整回归 **207 passed，0 failed，0 errors，0 skipped**，机器记录 [remaining-validation.json](remaining-validation.json) 保留此前 180 项结果的范围，并明确区分早前 CRANN CLI 记录和当前完整测试。

## commit6：接入全部地图匹配算法

新增 STMatching 与 IVMM 的无权重直接评估入口，与 HMMM 共用原子路网 DataBundle、路线评估和 JSON/CSV 导出，三种地图匹配算法均完成迁移。每条轨迹创建独立算法状态，修复旧 run 将上一条轨迹候选集合累加到下一条的问题；两算法修复尾部空候选时先索引再检查界限的越界错误。

验证：新增 **12 项通过**，包括两种算法有/无时间戳的多轨迹评估，逐轨迹对照原算法，重复运行的状态隔离，尾部无候选、全部无候选和单点轨迹，以及实际 CLI 和拒绝训练/续训/检查点请求。代表路网上 RMF=0、AN=1、AL=1。未初始化优化器、TensorBoard 或检查点。迁移表为 29/100，71 条 pending。

限制明确保留：原算法 temporal_analysis 是空实现，本轮沿用其空间/投票数学逻辑，时间戳保留用于输出，不声称已实现论文中的额外时间速度评分。

## commit7：迁移 FPMC 个性化轨迹模型

新增标准 nn.Module，保留 UI/IU/LI/IL 四个嵌入矩阵的偏好和一阶转移分解；显式 lengths 选择最后有效位置，移除旧 Batch.get_origin_len 与模型内部 device。个性化轨迹适配按 .usr 建立用户映射并验证未知用户，无 .usr 时从轨迹构造稳定词表；基础 RNN 路径不强制增加用户字段。

修复旧 calculate_loss 已注释承认的 NLL 错误：未归一化分数不能直接送入 NLLLoss。现在去除 padding 类后返回 log_softmax，目标/导出与公共零基位置词表一致，数学上等价于对原分解分数做全词表交叉熵。

验证：新增 4 项，覆盖旧四矩阵分数和正确交叉熵梯度、概率归一化、用户词表校验、batch=1/float64/填充隔离/无标签前向，以及完整训练、独立全词表排名导出和精确续训；与既有跨任务测试合计 **17 passed**。迁移表更新为 30/100，70 条 pending。

前七批完成后的完整回归为 **223 passed，0 failed，0 errors，0 skipped**，见 [完整机器记录](remaining-validation.json)。在仓库外新建独立数据/运行目录实际执行 CRANN 和 FPMC 各自训练、独立评估和续训共 **6 条 CLI 命令**，均成功，续训达到第三轮，独立评估与训练后评估一致；CRANN 训练缓存 miss，评估/续训均 hit，见 [当前 CLI 记录](remaining-cli-validation.json)。

## commit8：迁移 ChebConv/GAT 节点属性重建

保留两模型的属性自编码目标，新增具名 road_reconstruction 协议、轻量原子属性适配和公共训练组件下的损失/评估。模型预设不能改写受保护的协议字段；协议及相对默认任务的差异显式记录，避免将 LINE 的边采样目标冒充节点重建。模型为标准 nn.Module，无 scaler/logger/device，无前向文件写入；静态图注册 buffer，统一在评估阶段导出嵌入及节点 ID。

数据以完整已知图/属性为输入，训练、验证、测试监督节点互斥；仅训练节点拟合逐特征统计量。常量列尺度回退为 1，缺失数值输入用训练均值填补，原始 NaN 目标不计入损失。图使用只读 2×E 边表与稀疏算子，避免构造 N×N 稠密邻接矩阵。保持传导式属性重建语义，不宣称下游分类或归纳泛化认证。

修复：ChebConv 的双图高阶递推错误地沿用上一图的状态，以及 0 阶分支；稀疏乘法在 AMP 中保持匹配的 FP32/FP64 运算，稠密投影仍使用 autocast。GAT 按目标邻域与头分别做稳定 softmax，支持空边集并修复混合精度聚合类型。自动设备选择尊重模型能力，ChebConv 在仅有 MPS 的机器可回退 CPU，显式不支持的设备仍报错。

验证：新增 **12 项通过**，包含旧前向/梯度、双图独立多项式与 0 阶、完整训练/独立评估/精确续训及嵌入导出、归一化统计与缺失值/所有权、float64/CPU AMP、空图、协议指标校验、2,000 节点/4,000 边稀疏前向和设备选择。完整回归 **235 passed，0 failed，0 errors，0 skipped**。从仓库外执行两个发布夹具生成命令及 6 条训练/评估/续训 CLI，独立评估一致且续训到第三轮，见 [道路重建 CLI 记录](road-reconstruction-cli-validation.json)。迁移表为 32/100，68 条 pending。

## commit9：迁移 ASTGCN/MSTGCN 三周期预测

新增标准 nn.Module 三周期模型，通过元信息将 trend/period/closeness 输入明确映射到周/日/小时分支，保留时空注意力、切比雪夫图卷积、残差/层归一化及矩阵融合数学逻辑。图谱基注册 buffer，临时张量随输入设备/类型创建，移除模型内 scaler/logger/device。复用公共训练、原尺度掩码与检查点，未操作延期 Common 变体。

修复：MSTGCN 后续块把图过滤器宽度当成时间过滤器输出宽度，在二者不相等时维度错误；图谱归一化对单节点/零拉普拉斯图除零。谱基构造不再依赖随机特征值迭代；图卷积沿时间轴批量计算。周期采样差异写入 protocol_changes，摘要增加 history_span_points，避免把输入样本数误认为实际历史跨度。

验证：新增 **8 项通过**，覆盖三分支旧前向/梯度对照、明确周期顺序、完整训练/独立评估/精确续训、不同过滤器宽度与多目标、batch=1/float64/无标签和输入所有权、CPU AMP 与单节点图。完整回归 **243 passed，0 failed，0 errors，0 skipped**。两个发布夹具及仓库外 6 条 CLI 生命周期命令通过，续训到第三轮，见 [周期模型 CLI 记录](multiscale-cli-validation.json)。清单为 34/100 已迁移，66 条 pending。

## commit10：迁移 MegaCRN 记忆与课程训练

新增标准 nn.Module 与记忆辅助训练 step，保留自适应双图、编码/解码、记忆查询、分离与紧凑性正则。普通 forward 始终返回预测，不读取标签或依赖训练态返回类型；归一化目标和 batches_seen 由训练步显式传入。解码仅接收声明的已知 future_features，临时隐状态按输入设备/类型创建；公共检查点恢复课程计数和 NumPy 随机采样。

修复：旧调用在无标签或缺 batches_seen 时仍尝试课程学习；旧图卷积单位阵固定 FP32 导致 float64/混合精度类型不一致；长训练的 exp 课程阈值可能溢出。现在无标签 forward 不消耗课程 RNG，显式训练辅助路径执行采样，阈值在极大步数安全归零。验证使用纯预测指标，辅助记忆正则只参与训练。

验证：新增 **4 项通过**，覆盖旧预测、记忆输出、训练目标和逐参数梯度对照，纯 forward 的标签/RNG 隔离，三轮课程训练/独立评估/中断恢复且 NumPy 状态、计数、调度器及每个参数一致，无协变量/batch=1/float64，以及 CPU AMP。完整回归 **247 passed，0 failed，0 errors，0 skipped**。发布夹具和仓库外 3 条 CLI 训练/评估/续训命令通过，续训第三轮，见 [MegaCRN CLI 记录](megacrn-cli-validation.json)。清单为 35/100，65 条 pending。

## commit11：补齐无卷积模型的 CPU AMP 反向回退

新问题：TESTAM 实际 CPU BF16 训练在矩阵反向中触发同样的 oneDNN 不支持错误。此前只为含 Conv 模块的网络启用回退，遗漏了仅含线性层、注意力和矩阵运算的模型。

修复：隔离 RNG 的一次性后端探测同时执行卷积与线性投影前向/反向；所有 CPU AMP 训练按实际探测结果选择反向内核，不再按模型中是否存在 Conv 决定。仍只在反向上下文切换后端，结束时恢复设置，保留混合精度前向，CUDA 路径不受影响。

验证：CPU AMP、定制训练与 STSSL 共 **13 passed**。新增无卷积 STAEformer 实际训练与回退选择验证；原卷积模型的连续/中断恢复和后端设置恢复保持通过。TESTAM 迁移与完整回归在独立提交中验收。

## commit12：迁移 TESTAM 三专家与路由监督

新增标准 nn.Module、训练期辅助前向与简短 expert_step，保留时间专家、动态图专家、注意力专家及记忆门控。普通 forward 始终返回预测，不读取标签。原始标签、专家误差、分位数与非分位数路由监督由训练 step 处理，验证使用公共预测指标。公共工具新增显式 inverse_sqrt_cosine 步级调度器，保留旧预热/周期学习率曲线，状态由统一检查点保存。

修复：旧自适应专家把隐藏残差通道作为预测；旧路由按 input_dim 展平，破坏多观测/多目标；未来时刻固定加 12，且假定时刻总在最后一列；非分位数监督沿错误轴索引；未来时间专家的默认零输入在观测通道多于隐藏宽度时维度错误。现在按专家轴选择预测通道，时刻/观测索引来自元信息，预测时刻按实际窗口及已知 future_features 计算，目标零值和 NaN 同时从专家与路由监督中排除。当前明确要求连续输入且 output_window<=input_window，取尾部对应未来的预测槽位，不静默改变评测窗口。

验证：新增 TESTAM **8 项通过**，覆盖原专家计算、修正后的路由与辅助目标及梯度、prob_mul 两种分支、两种路由训练模式、原学习率曲线、观测通道数大于隐藏宽度、多目标/batch=1/float64/无标签、原尺度缺失值掩码、CPU AMP，以及三轮 dropout 连续训练与中断恢复逐参数、步数和调度器一致。完整回归 **256 passed，0 failed，0 errors，0 skipped**。发布夹具和仓库外训练/评估/续训共 4 条命令成功，独立评估与训练后评估一致，续训至第三轮，见 [TESTAM CLI 记录](testam-cli-validation.json)。迁移表为 **36/100**，64 条 pending；按本期排除 7 个延期 Common 和 2 个模板的口径，仍有 **55 个**实际模型待迁移。

## commit13：修复 CPU AMP 循环网络的前向工作区

新问题：ToGCN 的真实 AMP 训练中，LSTM 即使在 backward 时关闭 oneDNN，仍试图使用前向已创建的 BF16 oneDNN 工作区，报不支持反向。仅对反向切换内核不适用于这种循环网络。

修复：公共训练上下文在 CPU AMP 后端不支持且模型含 RNNBase 时，同时为训练前向选择原生循环内核；AMP 保留，后端设置在每个上下文结束后恢复。纯线性/卷积模型仍沿用此前回退策略，CUDA 不改变。

验证：公共 CPU AMP、定制训练与 STSSL 共 **14 passed**；新增两层 dropout TrafficLSTM 的实际 AMP 连续训练/中断恢复，下一轮参数与计数逐项一致，后端设置恢复。另加新模型批次共 28 项通过，新模型在下一提交独立验收。

## commit14：迁移 ToGCN 与 SimST 图时序预测

ToGCN 保留图卷积、LSTM 编码/解码、历史预热和整批 teacher forcing。普通 forward 不读 y、不消耗 teacher RNG；训练目标由公共 teacher step 显式传入，未来额外输入只取已知协变量，其他观测沿用历史。静态邻接注册 buffer，输出恢复明确的节点/目标维，不再依赖 y.shape 确定预测窗，也不在构造时改变全局 anomaly 状态。

SimST 保留因果时间注意力、节点嵌入、出/入方向指定邻居与邻域均值。邻域静态投影注册 buffer，向量化生成特征；按元信息将星期转换为原 weekday/7。修复位置编码/因果窗误用 output_window、邻域特征固定 12 步、batch>64 隐式改变节点维及回归头忽略多目标的问题。全节点批次不依赖 node_idx；按显式 data.graph 选择图，不再隐式使用旧 Dataset 的 ID 重排与距离变换。

验证：新增 **14 项通过**，包括两种 ToGCN teacher/自回归前向与梯度对照、两种 SimST 邻域配置的独立输入参考和旧网络梯度、两模型训练/独立评估/精确续训（含 dropout 与 Python teacher RNG）、不同窗口、多目标/batch=1/float64/输入所有权/标签隔离、CPU AMP、65 样本批次一致性、无节点嵌入、因果 mask 和孤立节点方向性。完整回归 **271 passed，0 failed，0 errors，0 skipped**；仓库外两个发布夹具及 6 条生命周期命令成功，独立评估一致且续训到第三轮，见 [CLI 记录](graph-temporal-cli-validation.json)。迁移表为 **38/100**，62 条 pending；本期实际模型仍有 **53 个**待迁移，最终全量验收尚未完成。

## commit15：迁移 STNN 与 TGCLSTM 潜因子/图过滤循环预测

STNN 修复固定 12 步/41 节点、关系聚合混排时间/节点、解码 reshape 混排节点/预测步，以及 predict 原地改写 batch 的问题。关系和元信息只读，聚合使用明确的时间/节点轴；三种关系模式均接入公共训练。旧 ceil 整数索引令输入映射 ffunc 没有梯度，默认改用可微双线性因子插值，验证映射权重在实际训练中更新。这是有意算法变更；lookup=hard 可显式选择用于量化对照，其不可微性仍明确登记，不将完整新预测与旧混排实现声称为等价。多变量映射只生成真正使用的两个坐标，移除未使用的坐标输出。

TGCLSTM 保留逐步图过滤、门控细胞与滑动窗口重编码，所有静态过滤器注册 buffer，状态随输入设备/类型创建。预测步由配置指定，不读 y，不再从 y 取未来额外观测。FFR 通过轻量原子适配显式选择列，默认 FFR_20min，支持其他列或 null 普通图模式；按 .geo ID 顺序映射，缺失边补 0，修复旧 inf×0。源文件变化触发基础缓存失效，FFR 元信息进入续训检查。训练保留原 MSE、mask_value=null 的有限零目标语义，评估继续遵循统一任务协议。

验证：新增 **21 项通过**。STNN 对照独立逐时/逐节点计算和 grid_sample 插值及梯度，覆盖固定/refine/discover 与 hard/linear，原 12×41 因子硬选择一致，训练映射权重实际更新；TGCLSTM 四种 FFR/普通图、clamp/非 clamp 组合对照旧预测、目标与全部梯度。两模型均验证多目标/batch=1/float64/无标签/输入不变、完整训练/独立评估/精确续训、CPU AMP；另验证 FFR ID 顺序、缺失边零值、不可变元信息、缓存失效和错误提示。完整回归 **292 passed，0 failed，0 errors，0 skipped**。两个仓库外发布夹具与 6 条生命周期命令成功，续训到第三轮且独立评估一致，见 [CLI 记录](latent-recurrent-cli-validation.json)。迁移表为 **40/100**，60 条 pending；本期实际模型仍有 **51 个**待迁移。

## commit16：迁移 STTN 并修复前向图状态漂移

新增标准 nn.Module，复用图卷积、空间/时间注意力与融合门，公共训练/评估/检查点负责生命周期。静态邻接图注册 buffer，可学习空间嵌入独立复制；每次归一化原图而非覆盖 self.adj_mx，修复重复 forward/评估导致图逐步变化和检查点未保存该状态的问题。单节点图给出有限零中心结果；临时时刻索引随输入设备创建，移除内部 device 属性；按时刻收集结果后一次 stack，替代空 Tensor 与循环 cat。位置容量从采样元信息推导并至少覆盖输入窗，可显式指定并校验。

验证：新增 **5 项通过**，对照独立原首轮归一化公式、原网络预测与全部梯度；两层 dropout 训练、独立评估和中断续训参数/调度器/计数完全一致；单节点与三节点多目标、batch=1/float64、无标签和 RNG/输入隔离、多次调用全部 state_dict 不变、CPU AMP、位置容量错误提示。完整回归 **297 passed，0 failed，0 errors，0 skipped**。仓库外发布夹具及 3 条训练/评估/续训命令成功，独立评估一致并续训第三轮，见 [CLI 记录](sttn-cli-validation.json)。清单为 **41/100**，59 条 pending；本期实际模型仍有 **50 个**待迁移，最终全量验收尚未完成。

## commit17：迁移 TimeMixer 并补齐全部预测分支

新增标准 nn.Module，保留多尺度季节/趋势混合、逐尺度预测及残差融合，使用公共训练闭环。归一化上下文作为局部值显式传递，避免旧 mean/stdev 临时属性被其他前向覆盖；按明确节点轴恢复四维预测。输出只含目标维，修复独立通道模式把 feature_dim 错乘进 reshape；标量独立通道保留原来的直接预测头，非独立模式保留逐尺度输出头的偏置累积语义。

修复 DFT 原先沿隐藏通道而非时间做 FFT、误清零第一个样本、固定使用 5 而忽略 top_k、全局频率阈值以及奇数长度逆变换的问题。现在按样本/隐藏通道选时间频率，排除 DC；AMP 下仅 FFT 用 FP32。avg/max/conv 分别使用真实尺度长度，支持卷积奇数长度、零降采样层；移动平均偶数核使用保长左右填充，过度池化提前报错。删除实际未使用的时间/位置嵌入、PDM 层及 label_len/freq/embed 配置，避免提供无效开关。

验证：新增 **18 项通过**，包括 avg/max/conv 下 9 种原可用组合的旧前向/梯度对照、两种通道模式的 FFT+归一化+卷积训练/独立评估/dropout 精确续训、独立时间正弦频率与 top_k 参考、显式归一化上下文、奇数尺度/偶数核、多观测/多目标/batch=1/float64/无标签/无状态修改、CPU AMP FFT 与零尺度。完整回归 **315 passed，0 failed，0 errors，0 skipped**。仓库外发布夹具及 3 条生命周期命令成功，评估一致并续训第三轮，见 [CLI 记录](timemixer-cli-validation.json)。迁移表为 **42/100**，58 条 pending；本期实际模型仍有 **49 个**待迁移。

## commit18：迁移 STG2Seq 并修复图/注意力轴混排

保留六层长期、三层短期图卷积、时间注意力、逐目标通道注意力及滚动预测。普通 forward 不读 y，训练通过独立 teacher step 提供归一化目标，注意力查询仅取声明的 future_features。图谱基只注册一份 buffer，显式传给图层；沿用公共有限谱基构造，修复 1 阶和退化图。输出头按实际目标数构建，不再限制 1/2 通道。

原 ConvST 的时窗 reshape 和卷积输出 reshape 会混排时间、节点、通道；两种 Attention 的 reshape 还混排 batch 与时间/通道。现在用明确轴排列和收缩公式，去除这些混排。独立逐轴公式用来验证正确数学运算，不把新预测与旧错误排列声称为数值等价。时窗限定为 1<=window<=input_window；可没有未来协变量，未知未来观测不会从标签补入。训练 MSE/有限零值目标与公共评估口径分别明确保留。

验证：新增 **9 项通过**。独立逐时/逐节点/逐通道图卷积与注意力的数值和全部梯度对照，覆盖 1/3 阶；完整深度训练、独立评估与 teacher/BatchNorm 参数和计数精确恢复；三目标、多观测、单节点、batch=1/float64/无标签/输入不变、显式 teacher 与未来特征、CPU AMP。完整回归 **324 passed，0 failed，0 errors，0 skipped**。仓库外发布夹具及 3 条生命周期命令成功，独立评估一致且续训第三轮，见 [CLI 记录](stg2seq-cli-validation.json)。清单为 **43/100**，57 条 pending，本期实际模型仍有 **48 个**待迁移。

## commit19：拒绝使用修改后的原始数据续训

新问题：实际构造 GRU 一轮检查点后，将 .dyna 的 speed 全部增加 1000，旧入口仍可从该检查点继续第二轮，并保留原数据的 best_validation。此前只校验元信息/布局/归一化状态，未记录数据或图来源版本；使用 none 归一化时尤其容易漏检。

修复：训练上下文建立时记录原子输入文件和元信息显式文件引用的名字、大小、纳秒修改时间；包含嵌套与外部引用，不读取内容做哈希。该快照随上下文保留，last.pt 保存快照而不是每轮重新读取文件版本。续训在载入模型/优化器和执行更新之前核对，拒绝数值、图或来源集合发生版本变化的情况。数据目录内使用相对名字，copy2 保留元信息后可移动数据目录，缓存/运行路径仍可改变。

完整检查点格式升至 **3**；旧完整格式缺少来源信息，明确拒绝续训，纯 best.pt 评估保持有效。该检查是文件元信息版本检查，不是内容相等证明；单独修改 mtime 也会保守拒绝，刻意同时保持大小/mtime 的内容变化无法据此识别，未引入文件内容哈希。

验证：新增 **7 项通过**，包括真实数值和 STGCN 静态图变化后在任何更新前拒绝、mtime 单独变化、保留文件版本后迁移目录并精确恢复下一轮、嵌套/外部引用和缓存文件排除、旧格式拒绝且纯权重评估有效、训练过程中源文件改变后检查点仍记录原上下文快照。与训练引擎/定制训练/STSSL 合计 37 项通过；完整回归 **331 passed，0 failed，0 errors，0 skipped**。仓库外实际发布夹具、训练/评估/续训及修改数据后拒绝续训共 5 条命令按预期完成，见 [来源检查 CLI 记录](source-resume-cli-validation.json)。迁移范围仍为 **43/100**，本期 **48 个**实际模型与最终全量验收继续实施。

## commit20：迁移 DeepMove 历史轨迹注意力

新增标准 nn.Module 与历史轨迹 DataBundle 适配。历史只取当前行程开始前已结束的同用户有效行程，重叠行程不提前进入历史；按完成时间建立用户 Tensor 银行，样本共享前缀视图，collate 拷贝保证输入所有权。无历史样本使用零上下文，不再通过删除首条行程改变基准样本集合。公共轨迹入口同时校验非空唯一位置词表与有效时间。

修复旧历史 padding 参与 softmax、预测受同批其他行程长度影响的问题；支持显式长度、全空历史和无效填充值隔离。RNN 分支使用正确 RNN 解码器，补齐旧未完成的 concat 注意力并初始化其参数，移除未使用的 fc_attn。仅对最后当前状态做注意力，去除重复计算；默认隐状态随输入设备/类型创建，forward 不读标签，输出去除 padding 类后的全词表 log_softmax。

验证：新增 **14 项通过**，包括 GRU/LSTM×dot/general 四种旧可用非填充分支的条件词表分数、NLL 与梯度对照；RNN/GRU/LSTM 三种训练/独立评估/导出/dropout 精确续训；历史/当前 padding、混批与独立推理等价、全空历史、batch=1/float64/无标签、concat 独立分数、基准样本 ID 保留、历史严格时间与重叠行程、collate 所有权、三种循环单元 CPU AMP。完整回归 **345 passed，0 failed，0 errors，0 skipped**。仓库外发布夹具与 3 条生命周期命令成功，独立评估一致并续训第三轮，见 [CLI 记录](deepmove-cli-validation.json)。示例生成器按注册信息选择任务，不再把新命名模型默认为交通预测。迁移清单为 **44/100**，56 条 pending，本期仍有 **47 个**实际模型待迁移。

## commit21：迁移 STRNN 并消除真实目标时空输入

原 predict 读取真实 target_tim 和按真实目的地计算的 current_dis，用这一目标条件向量给所有候选打分；源码已注释承认应逐候选建向量。现在保留原上下界时间/空间矩阵、全局项及用户/位置嵌入，每个候选分别计算几何状态。默认查询时刻为最后一次已知观测，模型可显式接受预先已知的 query_offset_hours，不从真实下一事件时间生成输入。elapsed_hours 仅含当前前缀，保留亚小时精度，自动时间边界只拟合训练前缀；距离使用静态 POI 的欧氏坐标度，中心化后注册 buffer，边界从包围盒推导并允许显式覆盖。零跨度有有限回退，超界比例夹紧，避免旧 LongTensor 上界截断/除零；坐标用 JSON 解析，不再依赖旧 eval。

相同历史位置先聚合，再按候选块计算，避免 B×V×L×H 中间量；padding、查询偏移与位置词表有显式校验。forward 始终返回去除 padding 类的全词表 log_softmax，公共 NLL 替代旧对整个 batch 的正样本总分做 logistic 的损失，消除样本耦合并加入分类负类监督。上述为有意修复，不与旧错误 predict/loss 声称等价；原转移矩阵本身按每个候选对照。

验证：新增 **8 项通过**，包括原转移矩阵逐候选分数、NLL 与全部梯度；训练/独立排名评估/导出/精确续训；分块、单独/混批、padding、batch=1/float64/目标字段隔离、几何状态的独立解析公式、训练边界与前缀时间核对、修改下一事件时间后输入/预测不变、重合坐标/零时间跨度、坐标错误与 CPU AMP。完整回归 **353 passed，0 failed，0 errors，0 skipped**。仓库外发布夹具及 3 条生命周期命令成功，评估一致并续训第三轮，见 [CLI 记录](strnn-cli-validation.json)。迁移表为 **45/100**，55 条 pending，本期仍有 **46 个**实际模型待迁移。

## commit22：迁移 HSTLSTM 并修复记忆单元与分桶泄漏

原 ST-LSTM 的 next_hidden 使用 out_gate*tanh(cell_gate)，而非 out_gate*tanh(next_cell)，导致更新后的记忆未正确参与输出。现保留位置与时空嵌入门控，修正该递推；padding 步不更新隐藏/记忆，只保存最后有效状态，避免堆叠整个序列的两套状态。模型为标准 nn.Module，不持有 device/scaler/logger，推理不读取标签；公共循环负责 NLL、全词表排名、选模、导出和完整续训。

新轻量数据适配只取当前已观察前缀，时间间隔为秒、距离保留 WGS84 测地线千米，首点仍为 0 秒/0.5 千米。原编码器在全量数据上计算分桶上界，现仅在训练前缀拟合，保留旧最小边界与向上取整；验证/测试超界夹紧，不重新拟合。桶作为 buffer 随模型迁移，保留浮点边界，用向量化 searchsorted/插值替代逐元素 Python 查桶、tolist 和固定 FP32/device 转换。仍对应仓库单层分类实现，不声称另行完成论文的额外层级结构；记忆公式修复和浮点分桶为有意变化，不宣称与旧错误计算逐值等价。

验证：新增 **10 项通过**。覆盖独立门控/记忆计算和全部梯度、显式修正旧记忆公式后的原分类前向/梯度、浮点槽边界/插值/夹紧、WGS84 原子前缀核对、修改真实下一事件时间后输入/预测不变、测试前缀超远距离不扩大训练桶、collate 所有权、padding/混批/batch=1/float64/无标签、零间隔/重合坐标、完整训练/独立排名评估/导出/逐参数精确续训、CPU AMP。完整回归 **363 passed，0 failed，0 errors，0 skipped**；仓库外发布夹具和 3 条生命周期命令成功，独立评估一致并续训第三轮，见 [CLI 记录](hstlstm-cli-validation.json)。迁移清单为 **46/100**，54 条 pending；排除延期 Common 和模板后，本期仍有 **45 个**实际模型待迁移。

## commit23：迁移 ATSTLSTM 并分离目标、负采样与候选几何

原 forward 将真实 target_loc/target_dis/target_tim 与负样本拼入预测路径，负样本时间也取真实下一事件；注意力没有屏蔽 padding，导致相同前缀受同批长度影响。现标准 nn.Module 只读取已知位置、相邻间隔、用户及候选几何，forward 返回不含 padding 的全词表原始分数。候选距离从最后已观察位置分别计算，查询间隔默认 0，只允许调用者显式提供预先已知的 query_offset_seconds。首点保留 100 秒/1 千米，其他输入使用秒及 WGS84 测地线千米；填充在嵌入前隔离，并在注意力 softmax 前掩码。

保留归一化成对排序目标，不替换成 NLL。具名 atst_ranking 协议声明 normalized_bpr 与 negative_samples；热度只统计训练时间区间原子点，按热度每批无放回重采样并排除正类，请求数按可用训练位置数减一截断。训练步显式选择正/负候选，按样本平均负对损失以适配公共梯度累积，使用稳定 softplus 计算原 logistic 目标；评估不随机采样，对全词表计算确定性 BPR 和公共排名。上述与旧固定且可能包含正类的采样、样本候选评估及 batch 总和标度存在有意差异，协议写入运行摘要，不能把 BPR loss 当 NLL 对比。

数据层复用 HSTLSTM 的已观察前缀计算，模型保持独立的首点约定。候选投影做代数分解，不构造 B×V×H；训练仅计算所选候选分数。每个组批器最多缓存 16 个起点距离行，避免 V×V 表，输出拷贝防止缓存污染。静态 WGS84 距离采用向量化 Vincenty，极点/未收敛点对通过已有 geopy/Karney 回退；全球随机及对跖等边界与 Karney 核对，不引入新依赖。

验证：新增 **11 项通过**，包括旧有效非填充分支的候选分数、原归一化 BPR 和全部梯度；候选子集训练与全量分数的损失/梯度一致且禁止调用全量 forward；修改真实下一位置/时刻后输入与预测不变、原子训练热度核对、正类排除/无放回、确定性评估、注意力 padding/混批/batch=1/float64、有限缓存命中与所有权、协议非法配置、CPU AMP、spawn worker，以及三轮完整训练/独立评估/排名导出/含随机负采样的精确续训。HSTLSTM 复用前缀后的 10 项验收保持通过。完整回归 **374 passed，0 failed，0 errors，0 skipped**；仓库外发布夹具及 3 条生命周期命令均成功，评估一致并续训第三轮，见 [CLI 记录](atstlstm-cli-validation.json)。迁移清单为 **47/100**，53 条 pending，本期仍有 **44 个**实际模型待迁移。

## commit24：迁移 SERM 并保留预训练语义权重

旧 SERM 在载入词向量后调用全模型初始化，导致预训练 Linear 权重被覆盖；前向构造 text_size 平方的单位矩阵、嵌套 NumPy 独热、CPU Tensor 和逐样本隐藏状态拼接。新标准 nn.Module 保留位置/时间/语义 LSTM 与用户偏好融合，初始化不再触碰语义权重，词向量仍可训练。静态 POI 词 ID 注册 buffer，用 EmbeddingBag 均值直接实现原有词频平均，重复词、空/OOV 类别均正确处理；直接读取最后有效 LSTM 隐状态，前向只有 Tensor 输入和预测输出，公共循环负责 NLL、全词表排名及完整续训。

旧编码器把真实 location 当作 DataFrame 行号，且数据集名称判断不一致、词向量路径硬编码并整文件入内存。新轻量适配与公共轨迹词表共用 geo_id 顺序，允许任何显式声明类别列的数据集；config.json 必须给出 word_vector_file，semantic_column 默认 venue_category_name，相对资源路径从数据集目录解析。按静态类别的小写空白词流式保留所需 GloVe 向量，不加载无关词入字典，全部找到后可提前结束；维度由文件读取，已用向量的维度/非有限数值、缺失资源有明确错误。摘要记录实际命中词、维度与空语义 POI 数。文件作为显式辅助来源进入现有格式 3 检查点，修改外部词向量后在更新前拒绝续训，不新增哈希校验。

验证：新增 **16 项通过**。预训练原值保留及可训练梯度、语义重复词/空类别的独立均值、同权重下旧独热均值/分类/全部梯度对照、重排 .geo 后的非连续真实 ID 对齐和仓库外路径、流式选择/提前结束、缺失或非法资源、全 OOV 类别、padding/混批/batch=1/float64/无标签与输入所有权、完整训练/独立排名评估/导出及 RMSprop/调度器/逐参数精确续训、外部向量只增加未用词也会在下一次更新前拒绝续训、CPU AMP。完整回归 **390 passed，0 failed，0 errors，0 skipped**。仓库外发布夹具和 3 条生命周期命令成功，独立评估一致并续训第三轮，见 [CLI 记录](serm-cli-validation.json)。预训练覆盖修复属于有意变化，不声称与旧错误初始化的训练结果相同；发布夹具是合成三维词向量，不是 GloVe 精度基准。迁移表为 **48/100**，52 条 pending，本期仍有 **43 个**实际模型待迁移。

## commit25：迁移 STAN 并修复目标时间、注意力与采样

旧候选时间向量使用真实下一事件时刻，编码器还把时间最大值记录到下界字段；自注意力 softmax 后再屏蔽 padding，导致有效权重受补齐长度影响。新标准 nn.Module 保留两级时空间隔注意力和位置融合，默认以最后已观察时刻查询，可显式提供预先已知的 query_offset_hours，不读取真实目标位置/时刻。每周小时为 1–168、真实用户从 1 开始，0 保留作 padding。时间上界仅拟合训练前缀，修正上下界顺序、夹紧超界比例，退化范围回退为 1；填充键先掩码再 softmax，填充查询/候选贡献归零。max_len 是可学习位置向量的容量，模型构建时验证覆盖全部前缀，不通过裁剪修改基准。

距离保留原半径 6371 千米的大圆定义。用静态球面反向点近邻计算空间上界，不构建 POI 平方距离表；重复坐标去重，发现亚米级点云会因反向欧氏距离舍入而低估毫米级直径后，补入固定大小分块回退与用例。中心化坐标注册 buffer；间隔嵌入随后求和的线性关系先行代数合并，移除 B×L×L/V×E 张量，候选按块计算。相同有效运算下的分数与全部参数梯度对照通过，不把旧错误掩码、上下界或目标时间输入声称为等价。

具名 stan_ranking 协议保留采样分类目标，training.loss.name=sampled_softmax；训练步去重批内正类并映射标签，从其余完整词表无放回抽取共享负类，数量按可用词表截断。修复旧循环在负类不足时反复重试、遗漏第 0 类负样本、同一正类在候选表多次出现的问题；批次覆盖全部类别时直接做全类 CE。训练只计算候选子集，标准 forward 输出全词表 log_softmax，公共验证/测试用确定性 NLL 与排名。移除模型自增 seed 和 Python 重播种，随机状态由公共检查点保存。

验证：新增 **15 项通过**，包括显式修复掩码并使用正确上界后的旧两级分数/全部梯度、静态直径对照及极点/对跖/毫米级范围、小时/用户编码与训练时间边界、修改真实下一位置/时刻后输入/预测不变、重复正类/全类别批次/第 0 类负样本、候选子集与全量梯度一致且训练禁止调用全量 forward、padding/混批/分块/batch=1/float64/无标签、单位置/重合坐标/零时间范围、非法坐标和位置容量启动前检查、CPU AMP，以及三轮训练/独立评估/排名导出/采样 RNG 与逐参数精确续训。完整回归 **405 passed，0 failed，0 errors，0 skipped**；仓库外发布夹具与 3 条生命周期命令成功，独立评估一致并续训第三轮，见 [CLI 记录](stan-cli-validation.json)。迁移表为 **49/100**，51 条 pending，本期仍有 **42 个**实际模型待迁移。

## commit26：无参数更新的轮次不提前衰减学习率（33f5f5f）

新问题：整轮无有效监督、无可构造负样本或 AMP 跳过全部更新时，epoch 调度器仍推进，导致首次更新前就衰减学习率并发出顺序警告。默认循环、MTGNN 和 STSSL 现都记录该轮开始的 optimizer_steps；只有该轮实际更新才推进 epoch 调度。验证、早停、日志、检查点及 metric 调度仍正常执行，不改变检查点格式。

公共引擎/定制训练/STSSL **31 项通过**。新增测试连续跳过两轮，确认学习率不变；从第二轮恢复后首次有效训练，与连续训练的参数/状态/调度器精确相同，且只衰减一次、没有顺序警告。详见 [问题与修复](empty-epoch-scheduler_zh.md)。后续 CARA 提交的完整回归同时覆盖此修复。

## commit27：迁移 CARA 并修复激活、候选与独立排序

旧包装器取最前 5 点，预测读取真实目标位置/时刻并随机生成 101 候选；训练先跨 batch 汇总分数，使样本耦合，负采样可能包含正类或在词表耗尽时无限重试。时间间隔按日历编码相减，距离还混用了经纬度与角度、负距离从错误的正类位置出发并截断为整数。新输入只包含已知前缀，按真实时间戳计算小时差；查询默认最后观察时刻，显式已知偏移可通过 query_offset_hours 提供，跨午夜/周末正确计算日历嵌入。坐标按 JSON/真实词表解析，用原半径 6373 千米的大圆小数距离，候选距离均从最后已知位置计算。

长序列探针进一步暴露旧门控不稳定；对照[作者官方实现](https://github.com/feay1234/CARA/blob/master/CARA.ipynb)及 [Keras 1.2 激活定义](https://github.com/keras-team/keras/blob/1.2.2/keras/backend/tensorflow_backend.py)，修复 LibCity 将门控与候选激活颠倒的问题，使用斜率 0.2 的 hard sigmoid 门控和 tanh 候选，并恢复作者的小正态嵌入与 Glorot uniform 矩阵初始化。其余官方连接保留。默认使用完整已知前缀；history_window 正值选择最近窗口，首步间隔/距离归零。先编码前缀一次，再逐候选执行一个单元，按块打分，移除 NumPy/CPU 往返和模型内部临时张量属性；共享坐标校验/球面距离函数供现有模型复用。

cara_ranking 协议明确 training.loss.name=bpr。每样本独立计算原成对目标及常数 1，再在批次取均值；负采样优先排除已访问/正类，无未访问位置时退回非正类，单位置无负对时跳过更新。验证/测试固定使用完整词表，报告所有非正类平均 BPR 和公共排名，不将 BPR 当 NLL 或把此协议声称为作者随机候选精度复现。采样 RNG、调度器和参数通过公共格式 3 检查点恢复；门控、输入与损失修复均为有意变化，不宣称与旧错误完整路径等价。

验证：CARA 新增 **12 项通过**，覆盖显式纠正激活后的作者兼容递推/候选分数/全部梯度、逐样本 BPR 与样本独立性、跨周末午夜的真实间隔/查询、修改真实下一事件后输入/预测不变、均匀负采样/词表耗尽/单位置零更新、候选子集与全量梯度且只编码一次、128 步有限前向/梯度与最近窗口对照、球面小数距离、padding/混批/分块/batch=1/float64/输入所有权、三轮训练/独立全词表排名/导出及采样 RNG/逐参数精确续训、CPU AMP。共享函数的 STAN/HSTLSTM/ATSTLSTM 验收保持通过。完整回归 **418 passed，0 failed，0 errors，0 skipped**；仓库外发布夹具及 3 条生命周期命令成功，独立评估一致并续训第三轮，见 [CLI 记录](cara-cli-validation.json)。迁移表为 **50/100**，50 条 pending，本期仍有 **41 个**实际模型待迁移。

## commit28：迁移 LSTPM 与因果历史行程适配

问题：旧时间槽相似度从全量行程统计，混入验证/测试位置；首行程被直接丢弃、空历史会在拼接时报错。空间最近父索引对每个增长前缀重新计算；批次内逐用户/逐点运行 LSTMCell，模型持有 device、CPU FloatTensor 和未注册的时间相似度。历史输入及距离为嵌套列表，旧 Batch 接口已经退役。

修改：接入标准 Tensor 字典和统一训练/排名生命周期。复用严格已完成历史银行，新增显式历史行程长度，保持跨历史行程连续 LSTM 状态、行程内部时间权重及两级注意力；对照[作者源码](https://github.com/NLPWM-WHU/LSTPM/blob/master/train/train.py)确认跨行程状态传递，并不将其误改为每条行程重置。历史一次展平打包运行，再按行程索引 scatter 聚合，避免展开行程×全部历史点的稠密权重表。空间父索引每条行程预计算一次，各前缀共享视图，逐时间步批量运行；用不可变状态列表保证 backward 正确，避免原地改写状态银行造成 autograd 版本错误。

时间相似度只在原子训练时间区间统计各槽已访问位置集合，保持 48 槽对角线及空槽约定，padding 行列额外为零。空间距离沿用原 LibCity 的 6371 km 球面、三位小数千米取整和最近先前点同距时选择较近时间索引；历史距离保持最后观察 POI 到算术行程质心、至少 1 km，不改成作者仓库的逐点平均距离。共享 NumPy 球面函数从 STAN 适配中移入公共静态点工具，避免 LSTPM 依赖另一模型的数据适配。历史只含当前行程开始前已完成的行程，空历史为零长期上下文，保留全部公共基准样本；时间/位置均不读取真实下一事件。

原有效路径前向和全部梯度对齐；有意变化为去掉 padding 类后使用公共平均 NLL（原为包含 padding 的 summed NLL）、训练统计去泄漏、首行程保留，以及移除从未被调用的 dropout 配置。时间相似度注册 buffer，临时状态随输入设备与类型，模型不持有 device/scaler/logger。发布 lstpm.json 和 --model LSTPM 夹具。补齐前表 commit27 与 c9b77ca 的映射。

验证：新增 **14 项通过**，覆盖原两级注意力/跨行程状态/膨胀递推/全部参数梯度，显式行程边界与质心距离、训练原子时间相似度/空槽、每个前缀空间父索引/同距规则、重叠行程排除/首行程保留、真实下一事件变更后输入与预测不变、完整训练/独立全词表排名/导出/精确续训、padding/混批/全空历史/batch=1/float64/输入与 buffer 所有权、错误输入、CPU AMP。完整回归 **432 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)；仓库外实际夹具、训练、独立评估、续训第三轮共 4 条命令成功，见 [CLI 记录](lstpm-cli-validation.json)。迁移表为 **51/100**，49 条 pending；排除 7 个延期 Common 和 2 个模板后，本期仍有 **40 个**实际模型待迁移，最终全量验收尚未完成。

## commit29：迁移 GeoSAN 并替换停止维护的 Field 接口

问题：数据适配依赖 TorchText 0.6 Field/NLTK，词频与访问集合读取全量未来数据；.geo 的经纬度顺序错置、BallTree 用角度而非弧度、近邻数超过实际位置数时报错，排除访问位置可能无限重试。旧模型使用九元组与 device、混合训练/推理候选布局；默认零宽度用户层、无用线性层和 Transformer 层原型造成未使用参数，非默认乘法/用户融合及位置单分支的维度不一致。

修改：新增标准库 QuadkeyVocabulary/字符串 n-gram，按 [TorchText 0.6 Vocab](https://github.com/pytorch/text/blob/0.6.0/torchtext/vocab.py) 与 [Field](https://github.com/pytorch/text/blob/0.6.0/torchtext/data/field.py) 冻结 PAD=0、无 UNK、频次降序及同频词典序规则，支持最小词频和张量 padding。旧数据参考源码同样移除 Field/NLTK，并将坐标 eval 改为 JSON；pyproject 删除 geosan 可选组，当前 editable 安装已实际重建。安装初次受沙箱带花括号的系统临时路径影响失败，指定工作区临时目录后成功，不把此环境路径问题改成项目代码。

新数据适配复用公共目标时间切分，保留所有基准前缀，不采用旧全量频次过滤及最后新位置留出。正确读取经纬度，静态 POI 建地理词表（默认 17 级/6-gram），仅训练原子观测建立已访问集合；位置表、词表、地理布局和用户/星期小时信息均显式记录。新模型为标准 nn.Module，静态地理词注册 buffer，保留原区域 Transformer/因果编码、两种位置编码与可选候选注意力的有效计算，普通 forward 只返回全词表原始分数。移除未用参数、模型设备与恒等注意力矩阵，区域位置相加不改写输入；显式源投影修正非默认用户/时间/乘法融合维度，位置单分支的候选也只用位置，正弦编码支持奇数宽度。

训练使用 geosan_ranking 具名协议和简短 sampled_step，复用公共外层训练，无需保留原独立 Executor。KNN 在正确纬度/经度弧度上查询，近邻数截断至词表大小，显式排除正类，均匀有放回采样并记录真实概率；过滤已访问耗尽时回退到非正类近邻，无无限循环。纯静态近邻缓存上限 128，不建立全 POI 两两距离；采样器按需初始化，独立评估不创建训练索引。加权二元目标保留 softmax 权重的梯度；训练仅打分所需候选，源编码在候选块间复用。验证/测试使用完整词表和确定性加权二元损失/排名；单位置无负类时保留正类训练项。与旧过滤/随机候选实验协议的差异明确登记，不将新全词表结果声称为旧实验精度复现。

验证：新增 GeoSAN **15 项**，含四种旧有效位置/解码组合的分数与全部参数梯度、冻结词表规则/Mercator 地理编码、加权二元损失和权重梯度、KNN 球面近邻/采样分布/缓存边界/访问耗尽、公共基准样本保留/训练统计隔离、真实下一事件变化后的输入隔离、因果/padding/混批/分块/float64 与候选子集梯度、非默认融合/用户/时间/奇数宽度、完整训练/独立评估/采样及 dropout RNG 精确续训、CPU AMP/单位置，以及阻止旧依赖导入后的实际训练。最后源码完整回归 **447 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。未安装 TorchText/NLTK 的环境中，仓库外发布夹具、训练、独立评估和续训第三轮共 4 条实际命令成功，指标一致，见 [CLI](geosan-cli-validation.json)；editable 安装和 pip check 通过。迁移表为 **52/100**，48 条 pending；轨迹预测全部 **13 个非模板条目**均已接入，本期仍有 **39 个**其他任务的实际模型待迁移，最终全量验收保持未完成。

## commit30：迁移 TTPNet 并补齐 ETA 模型覆盖

问题：旧 TTPNet 把所有路线长度设为批次最大值，填充点进入道路/速度 LSTM 和双向 LSTM；最终固定取批次最后位置。模型内部归一化、反归一化并读取耗时目标，输入/训练/推理职责混合；道路嵌入固定 16384×32 且直接用原位置值索引。编码器用 eval 解析列表、.seconds 丢弃天数、把星期一/日当周末，traj_len_idx 指到最后的 start_timestamp。速度输入没有出发前可用性约束；旧相对误差开平方在精确拟合时梯度为 NaN。

修改：按[作者模型](https://github.com/YibinShen/TTPNet/blob/master/TTPNet.py)与[速度网络](https://github.com/YibinShen/TTPNet/blob/master/Speed.py)保留现有 LibCity 的速度 CNN/RNN、道路/速度 LSTM、两层双向 LSTM、LayerNorm 与累计预测头，改成标准 nn.Module 和显式 Tensor 字典。所有路线递推按真实长度打包，短期三支统一批量计算，重复前缀求和改为一次 cumsum；每条路线按自身长度选择最终输出。道路向量按 .geo 的真实 ID 对齐，行数/维度不再硬编码，保留预训练值并可继续训练；普通 forward 始终只给整趟预测张量，辅助方法显式返回累计预测及掩码。

新增 ETA 共享完整行程读取器，DeepTTE/TTPNet 复用目标区间互斥切分、用户映射和路线几何/距离验证。TTPNet 的所有归一化和由速度/长度推算秒数均移至数据层，仅由训练行程拟合；km/h、m/s 与 km、m 显式换算，零速度保留原 0.2 km/h 处理。短期输入为每点 3×4×3，长期为 7×3，分别明确速度、长度和预计耗时通道。每点交通快照时间必须有效且不晚于该趟出发，缺失或未来快照直接拒绝；不把途中/到达后信息当作已知输入。实际累计耗时仅为 local_y_raw/y_raw 标签。静态图嵌入和历史速度由上游提供，不宣称补上了论文未在 LibCity 提供的离线张量分解训练器。

采用 ttpnet_eta 协议和公共 train_loop 的累计 step：保持训练逐路线累计相对 RMSE、验证整趟 MAPE，保留以训练 time_gap 统计量仿射输出的旧含义，逆变换放在 loss/评估。按有效点归一化后再按路线平均，填充标签不参与计算；用 vector_norm 使零误差处梯度有限，分母 eps 默认 1 秒。默认最短 3 点与 DeepTTE 默认样本一致，可显式 min_route_length=2；输出和 JSON 都标明秒。现有公共早停配置生效，patience=0 可关闭。公共 ETA 逆变换只把 FP16/BF16 提升为 FP32，保留 float64。旧参考编码器同步改用 JSON/total_seconds、修复周末和长度索引，不恢复旧执行接口。

数值对照中的初次 FP32 原始坐标归一化误差约 0.006689，导致预测差约 2.58e-6；定位为旧实现先截断大经纬度再相减的精度损失。新数据层先在双精度中归一化，模型数学对照使用双精度原始坐标后，最大预测差约 3.37e-10；没有通过放宽原检查阈值掩盖差异。全部模型参数梯度对照通过。其余有意变化（真实长度、ID 对齐、时间单位和输入可用性）明确登记，不将完整旧错误路径当等价基准。

验证：TTPNet 新增 **16 项通过**，覆盖旧有效非填充模型及全部梯度、按路线累计相对 RMSE/权重/梯度与精确拟合、训练/独立评估/秒级导出/dropout 参数与 RNG 精确续训、混批/变长/padding/batch=1/float64/所有权/标签隔离、完整行程共同基准/训练统计/预训练 ID 对齐/单位换算、实际未来耗时变化后输入与预测不变、非法/缺失/未来快照和速度 JSON、零速度/常量 CPU AMP、两点路线/非固定嵌入宽度、跨天总秒数和旧编码器修复。与共享任务/退役接口检查合计 **42 项通过**。当前源码完整回归 **463 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)；仓库外夹具、训练、独立评估、续训第三轮共 4 条实际命令成功，指标一致且预测按秒导出，见 [CLI](ttpnet-cli-validation.json)。迁移表 **53/100**，47 条 pending；两个 ETA 模型均已接入。本期仍有 **38 个**实际模型（35 交通、3 道路表征）及最终全量验收待完成。

## commit31：迁移 Node2Vec/DeepWalk 与明确外部训练能力

问题：旧模型只提供写全局缓存的 run，依赖 Gensim 顶层导入及旧 NetworkX/Python API；DeepWalk 的 start=0 会误选随机起点，默认 Random 对象在调用间共享状态。训练保存 Gensim pickle/.m，再从文本反读向量，缺少统一实验目录、真实节点 ID 校验和明确验证/续训能力。旧全图邻接矩阵、全量游走列表及全部二阶 alias 表也会造成较大内存开销。

修改：保留 Gensim skip-gram 后端，[Gensim 4.4](https://github.com/piskvorky/gensim/releases/tag/4.4.0) 支持 NumPy 2，embeddings 最低版本更新为 4.4，实际安装在当前 CPU 环境并通过 editable 安装/pip check。新 road_walk 协议完整定义五个配置区块，训练参数属于 Word2Vec，不继承无作用的神经训练项。ModelSpec 声明 kind=external、CPU、需要安全推理权重，但不支持验证和续训；轻量 AlgorithmContext 交给独立训练回调，不创建 PyTorch 优化器、调度器、GradScaler 或 TensorBoard。统一入口继续负责配置、日志、运行摘要和输出；CheckpointManager 增加集中式纯权重写出。既有神经及直接算法路径保持原职责，普通导入不加载 Gensim。

数据按真实 .geo ID 构建稀疏边表和行偏移；二值图默认按关系存在连边，Node2Vec 可显式读取正亲和权重。零边去掉、非法负/非有限权重拒绝，重复有向关系最后生效，无向化双向取最大，自环显式配置。全图无监督协议不创建伪 train/val/test loader。Node2Vec 的 p/q、加权 alias 采样和 DeepWalk 的均匀/重启随机游走保留；每个 corpus 使用局部 RNG，二阶缓存按 100,000 个邻居槽限制，概率归一化保持数值稳定。采样逐条写入实验 training/walks.txt，Gensim 按行重复读取，不把全部路径留在 Python 列表。旧参考代码同步修复 NetworkX 接口/无向标记、Iterable、起点 0 与共享默认 RNG，不恢复旧全局运行入口。

向量按节点索引直接从 Word2Vec 获取；Node2Vec 默认负采样 5，DeepWalk 保留 LibCity 的 hs=1 且 negative=5 组合，可显式选择仅 HS。运行种子、固定 token 初始化映射和单 worker 支持当前环境跨进程复现；确定性模式拒绝多 worker。单词初始化映射不涉及源文件内容哈希。训练统计/来源文件版本写入 training/report.json 和 metrics.jsonl，安全权重为 checkpoints/embeddings.pt（纯 Tensor，含节点 ID、算法配置、稀疏图与向量），不存储/加载 Gensim pickle。独立评估无 Gensim 依赖，严格拒绝不匹配的 ID/图/配置，按真实 ID 导出 embeddings.npy/node_ids.json；关闭导出会移除旧结果。

能力边界：没有验证划分或完整安全后端状态，因此无 best_validation、best.pt/last.pt 或完整续训；--resume 与调参在准备数据/训练前拒绝。结果只报告节点数/维度/覆盖率/均值范数/孤立及无出边节点等结构诊断，不能解释为模型精度；覆盖率 1 也不表示孤立节点有上下文训练信号。默认 epochs 沿用旧 1000，发布示例显式两轮，只证明生命周期和数值实现，不声称下游精度复现。

验证：新增 **17 项通过**，包括加权有向/无向 Node2Vec 固定游走对照、修正起点 0 后 DeepWalk 三种重启概率与重复调用、p/q 概率/alias 分布/对称权重/只读稀疏元信息、两个模型真实 Gensim 构造器逐向量完全一致、外部训练不创建 PyTorch 优化器或记录器、跨进程逐元素确定性、阻止 Gensim 导入后实际独立评估、导出关闭与 ID 对齐、明确能力拒绝、缺少训练依赖报错、损坏/不匹配权重拒绝、全孤立图与 2,000 节点稀疏适配/有界缓存。完整回归 **480 passed，0 failed，0 errors，0 skipped**（环境实际安装 Gensim 4.4），见 [汇总](remaining-validation.json)。仓库外两个夹具、两次训练及两次独立评估共 **6 条成功命令**，另有 **4 条续训/调参请求按预期拒绝**，见 [CLI](walk-cli-validation.json)。迁移表 **55/100**，45 条 pending；本期实际剩余 **36 个模型**（35 交通、1 道路 GeomGCN），最终全量验收仍未完成。

## commit32：迁移 GeomGCN 并完成道路表征覆盖

问题：旧 GeomGCN 顶层依赖旧 DGLGraph/子图 API，训练前向把中间向量写到全局缓存。其分区边集合已计算却未使用，后续构造节点诱导子图会混入别的分区边；子图结果未写回父图，读取父图时可能只得到零结果。第二层选择拼接时，嵌入/解码层仍按未拼接宽度构造，不能运行。旧 JSON 的 dropout 字段也未被实际读取（类读取 dropout_rate，缺省 0.5）。

修改：保留 LibCity 四层属性自编码器及实际使用的两分区（非自环道路边、每节点一个自环），没有把原论文的几何邻域构建作为新增范围。对照[作者的边分区聚合实现](https://github.com/bingzhewei/geom-gcn/blob/master/utils_layers.py)，用原生 PyTorch 投影/index_add 直接计算分区消息、完整有向入度的源/目标归一化、分区合并和多头合并。边和范数是 buffer，不持有 DGL 图、scaler、logger 或设备；前向不写文件、不改图，不读取目标。重复关系/输入自环不产生重复消息，每个节点显式恰好一个自环，孤立节点可以使用自身属性。

四个合并设置均支持 cat/mean，按实际宽度构建后续层；最终读出固定对分区/头取均值，保证属性重建宽度不变。这保留原默认及原可运行 mean 路径，同时为原本维度错误的拼接分支定义可用行为。embedding_dim 是每头每分区宽度，第二层拼接会增加实际导出维度。默认 hidden_dim=144、embedding_dim=32、dropout=0.5、无归一化及最终 ReLU 按旧有效配置保留；可显式使用仅训练节点拟合的 standard。公共重建损失修复无条件 float32 转换，只提升半精度，保留 float64 精度。

GeomGCN 复用 road_reconstruction 具名协议、公共数据适配/损失/评估和默认 train_loop，支持验证选模、独立评估与完整续训；向量和真实 node_ids 在评估层统一导出。迁移表不再给该模型要求 graph/DGL 可选组，旧源码仅作历史数学参考。发布 geomgcn.json 及 --model GeomGCN 夹具生成方式，同步更新运行指南、可选依赖登记和当前迁移数量。

验证：新增 **28 项通过**，包括全部 16 种分区/通道合并设置下修正后原公式的独立稠密前向、嵌入及全部参数梯度对照；诱导子图混边的专门反例、完整图入度、各层真实参数更新、阻止 DGL 导入后的训练/独立评估/嵌入与 ID 导出、dropout/优化器/调度器/早停/RNG/下一轮状态精确续训、无标签/全图与单节点子集/float64/输入和图状态不变、空图/单节点/自环去重、2,000 节点稀疏图、CPU AMP、缺失目标及双精度损失梯度。与既有道路重建/清单检查合计 **41 项通过**；之后当前源码完整回归 **508 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。仓库外实际夹具、训练、独立评估及续训第三轮共 **4 条成功命令**，独立指标/向量完全一致，见 [CLI](geomgcn-cli-validation.json)。

迁移表更新为 **56/100**，44 条 pending。道路表征全部 **6 个模型**已接入；排除 7 个延期 Common 和 2 个模板后，本期仍有 **35 个交通预测模型**及最终全量验收待完成，不能据此宣布全部主线完成。

## commit33：迁移 STResNet/ACFM 与原子网格数据

问题：旧网格输入/输出通过 view 混排空间与特征轴；STResNet/ACFM 的预测通道固定为 2，ACFM 注意力输入还硬编码为 2，使默认 lstm_channels=16 无法正确装配。旧网格数据与外部特征紧耦合，外部数据全时间轴归一化会使用测试区间信息；普通点数据入口不能完整读取 .grid，来源版本也未覆盖该格式。网格真实 1x1 单样本尾批的 BatchNorm 无法计算方差。

修改：新增具名 grid_cpt 协议和轻量原子适配，保留 STResNet/ACFM 的逐帧最近/日/周三段含义，与图模型的逐小时 CPT 区分。使用 .geo 的 row_id/column_id 和真实 geo_id 显式映射完整矩形，支持乱序 .grid 及 entity_id 型 .dyna，不分配无用途稠密图；切分、原尺度掩码、训练区间归一化、字典 Tensor 批次与公共评估格式复用既有契约。历史与未来外部输入分离，只允许日历和显式 known_future_columns 进入未来；外部观测不再按全时间轴归一化，也不把目标时刻真实外部观测自动当预测输入。缺失历史观测在输入侧以零填补，原始标签不变。

模型均为标准 nn.Module，网格排列使用 buffer；显式 permute 空间/特征/时间轴并恢复 [batch,horizon,node,feature] 输出，任意目标数与显式多步回归不再受固定 2 通道限制。STResNet 保留三分支残差、矩阵融合、外部网络及 tanh，删除未启用 BN 时的无用参数。ACFM 保留逐帧共享残差、逆序 ConvLSTM、两级注意力和时刻门控，按真实隐藏宽度装配注意力；支持纯 local、纯 global 及两者并用，历史和未来外部宽度可不同。五种原可运行残差模式有明确维度条件，未使用的 merge_mode 不保留。单样本 1x1 尾批使用 BN 运行统计，不复制样本或污染运行方差。

原始数组缓存继续使用公共原子缓存机制，增加网格命名空间、来源列表与字段/形状校验；窗口/切分变化可复用原始缓存，损坏数组会自动重建，历史输入不会越过缺失时间片或原子文件边界。来源版本加入 .grid，并识别外部显式 grid 文件，文件大小/修改时间变化时拒绝续训，没有源文件内容哈希校验。公共预测输入提取增加 history_features；公共 ForecastLoss 仅提升半精度，不再无条件丢失 float64 精度。发布两个配置、乱序原子网格夹具和 --model STResNet/ACFM 命令；Common 变体与深层数据缓存重构仍延期。

验证：新增网格 **43 项通过**，包括修正轴顺序后的旧 STResNet BN/外部分支与 ACFM 五种有效残差模式的预测/全部梯度对照、全流程训练/独立评估/BN/dropout/优化器/调度器/早停/RNG 精确续训、多变量/多步/单分支/batch=1/float64、原子 ID/行顺序/CPT 时间、目标区间互斥/训练统计、真实未来目标及未声明外部观测变化后的输入与预测隔离、原始缓存跨预测窗复用及缺失/损坏形状重建、时间缺口/文件边界、.grid 修改后续训拒绝、单节点尾批 BN/CPU AMP、无效模式与协议校验、双精度原尺度损失。当前源码完整回归 **551 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。仓库外两个夹具、两次训练、两次独立评估及两次续训共 **8 条成功命令**，评估指标/预测逐元素一致，续训第三轮且缓存复用，见 [CLI](grid-cli-validation.json)。

迁移表更新为 **58/100**，42 条 pending；排除 7 个延期 Common 和 2 个模板后，本期仍有 **33 个交通预测模型**及最终全量验收待完成。

## commit34：迁移 CONVGCN/MultiSTGCnet 与明确周期/距离语义

问题：CONVGCN 的旧数据代码将两个目标都取为通道 0，末尾完整窗口被丢弃；模型将图卷积投影时间与节点轴直接 reshape，混排 Conv3d 通道，历史长度还分别依赖数据 time_lag 与未正确装配的模型默认值。MultiSTGCnet 的最近与日周期都调用 daily_block，interval_block 未参与训练；距离分组会原地替换无穷值、把缺失边当零距离，并漏掉最大距离与常量行。静态矩阵没有统一作为 buffer，未启用的周期网络和未使用超参数仍被创建/登记。

修改：两模型均使用标准 nn.Module 和公共 train_loop、损失、评估、检查点及点数据原始缓存。CONVGCN 保留两层时间投影图卷积、节点通道 Conv3d 与全连接输出，显式排列投影/节点轴；邻接矩阵先复制后加单位自环并注册 buffer，图卷积偏置始终注册 Parameter。新增 convgcn_windows 协议，seasonal_windows 按 trend/period/closeness 拼接三个正序连续窗口，input_window=10 替代旧 time_lag=11 的隐式历史长度；服务周期 36/180 均是可配置原始时间片数，不冒充日历日/周。修复参考数据代码的第二目标通道，新适配直接读取所有声明目标并保留末尾完整窗口，任意特征/目标数及显式多步均可运行。

MultiSTGCnet 使用 multi_stgc 协议和 frame_cpt：最近帧、同刻日/周观测依次拼接，各段由近到远。最近、日、周分别调用 interval/daily/weekly 网络，只构建活动分支；没有最近段时省略空间分支，仍可用日/周预测。保留原三空间图分支、图乘法方向、两层 LSTM、融合及真实节点数×输入维度的循环宽度，未使用的 hidden_size/num_layers 不作为新参数。输出回归头/融合宽度显式涵盖全部预测步。

距离适配从真实 .geo ID 与 .rel 列构建有限列出边，distance_column/元信息 weight_col 选择原始距离，不将高斯相似度当距离。三个组按每行有限距离的等距值域划分，缺失边排除、显式零距离保留、最大距离包含、常量行归近组、孤立行全零；不修改输入矩阵。重复关系最后一条生效，对称化取两个方向已知最小距离，自环取零；可显式选择 binary 将列出边视为单位距离。矩阵/ID/列名进入只读元信息，矩阵作为 buffer，边数和边界语义写入摘要。距离列变化会重新准备缓存，源文件变更拒绝续训；不做文件内容哈希。点数据的完整时间跨度检查改用缺口前缀计数，保持原窗口判定并避免每个窗口重复扫描历史跨度。

验证：新增 **34 项通过**，包括原图投影/修正节点 Conv3d 轴后的前向及全部梯度、原循环数学/明确修复最近分支与距离支持后的前向及全部梯度、独立第二目标/服务周期窗口/末尾完整目标、距离最大值/常量/零距离/孤立图/无原地修改、活动周期与多步/多目标/batch=1/float64、完整训练/独立评估/优化器/调度器/早停/RNG 精确续训、静态矩阵/输入不变、距离列切换及来源续训保护、方向/对称化/二值单位距离、非法配置及 CPU AMP。暂停时剩余的测试数据类型错误通过显式将测试距离列转为 float 修复，没有降低检查标准。当前源码完整回归 **585 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。仓库外两个夹具、两次训练、两次独立评估及两次续训共 **8 条成功命令**，独立指标/预测逐元素一致、目标通道不同、续训第三轮且原始缓存复用，见 [CLI](periodic-cli-validation.json)。

范围：数值参考明确修正原错误轴顺序/分支/距离边界，不宣称与旧错误完整流程或论文精度等价。CONVGCN 的节点通道稠密卷积和 MultiSTGCnet 的全局节点特征 LSTM 仍保留原规模成本，未替换为另一个轻量网络；本轮不做大城市性能认证，Common 变体继续延期。迁移表更新为 **60/100**，40 条 pending；排除 7 个延期 Common 与 2 个模板后，本期仍有 **31 个交通模型**及最终全量验收待完成。

## commit35：迁移 DKFN 并修复跨样本方差与无标签滚动

问题：旧 DKFN 在融合中对整个 batch 计算 torch.var，同一样本会受其他样本影响并产生跨样本梯度；单元素/零方差会产生 NaN 或除零，未约束的系数也可能使分母失效。predict 读取 y 决定输出长度并从真实未来目标复制额外通道，纯推理不可用。图与过滤矩阵保存在普通列表/Variable 中，设备/类型迁移不完整，squeeze 还可能丢失标量/单批维度。

修改：标准 nn.Module 保留图过滤 LSTM、自身 LSTM、邻居细胞权重及[作者](https://github.com/Fanglanc/DKFN/blob/master/DKFN.py)的原系数融合式。图复制后二值化，保留原列度数左乘归一化、K 阶图幂及可选钳制，矩阵注册 buffer；隐藏状态随输入创建，不持有设备/scaler/logger，不使用 Variable。默认 hops=3、clamp_adjacency=false、threshold=1e-4，使用原 RMSprop 学习率 1e-4，直接复用公共 train_loop/损失/评估。

融合方差改为每个样本展平的观测/图特征流内计算，移除跨 batch 耦合；非退化流保留样本方差，单元素定义为零方差。物理系数 softplus(c) 初值 1，保存的 c 为无约束坐标；正性避免负系数导致异常分母，数值参考明确验证变换的链式梯度。两方差按共同尺度归一化，代数上保留作者非对称融合式而不另改成凸组合；半精度方差提升到 float32、float64 保留，两流方差均为零时回退为两个隐藏预测的均值。这些是有意修复，未将旧全 batch 方差错误路径当成等价目标。

forward 由配置决定预测长度，每步重新编码当前窗口；预测回填目标通道，未知额外观测保持最后值，只有已声明的 future_features 通道更新，不读取未来真实观测或 y；末步不构造无用后续窗口。模型/优化器状态（包括变换前系数）、RNG、调度器等由公共完整检查点恢复，独立评估只加载纯权重。发布 dkfn.json、--model DKFN 夹具命令及使用说明。ResLSTM 仍为 pending，将在独立批次处理。

验证：新增 **26 项通过**，包括 12 种阶数/钳制/输入输出维度的原递推与修正融合参考预测/全部参数梯度、作者融合公式/系数链式梯度、改变其他样本后的预测不变与跨样本梯度为零、图幂/钳制/元信息不变与 buffer 类型迁移、单节点/常量/孤立图/零方差/系数边界的有限预测及梯度、完整训练/独立评估/RMSprop/调度器/早停/RNG 精确续训、已知未来通道更新/无标签/输入及状态不变、CPU AMP 和非法参数。与清单检查合计 **27 项通过**，当前源码完整回归 **611 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。仓库外实际夹具、训练、独立评估和续训共 **4 条成功命令**，独立指标/预测逐元素一致、续训第三轮且原始缓存复用，见 [CLI](dkfn-cli-validation.json)。迁移表更新为 **61/100**，39 条 pending；排除 7 个延期 Common 与 2 个模板后，本期仍有 **30 个交通模型**及最终全量验收待完成。

## commit36：迁移 ResLSTM 并修复序列/池化/服务周期输入

问题：旧 ResLSTM 把 [batch,node,feature] 送入默认按时间优先的 LSTM，预测会受 batch 内其他样本影响；周期和帧直接 reshape 混排，池化不对称且展平层写死 node*96，奇数节点/其他窗口尺寸无法正确运行。输出始终只有一个预测步和一个目标，未下采样单元仍创建无用投影，图每次前向重建。数据只按总特征数取外部输入还可能把缺失流量/额外原始观测与日历混淆。完整历史跨度连续检查会把有夜间停运的服务周期数据全部过滤。

修改：标准 nn.Module 保留两流量残差 CNN、近期第一流量的图分支、可选历史外部网络、节点轴 LSTM 和线性乘性注意力。LSTM 明确 batch_first=True，周期/帧轴显式排列，flow_indices 选择两列原始流量；输出头涵盖配置的预测步和全部目标。SAME 池化在尾部补齐，展平尺寸按实际 ceil(nodes/2)*ceil(history/2) 推导；单元只在下采样时创建投影。图使用列度数两侧规范化，复制并注册 buffer，不使用 np.matrix、求逆或前向重建；隐藏计算不持有 scaler/logger/device。原 base_channels=32、recurrent_hidden=128、recurrent_layers=2 为默认，可显式缩小运行示例。

新增 reslstm_windows 协议，默认每段 5 帧、period_steps=72/trend_steps=360 原始服务帧，复用 seasonal_windows 和点数据缓存。轻量适配从数据元信息/原子表头识别原始列，至少要求两列真实流量；历史外部列仅来自原始观测之后的日历/.ext，额外原始列不会误作共享外部输入，也不读取未来 external/y。外部 LSTM 保留 LibCity 的实际串联形态，不声称是作者 Keras 外部分支的精确移植。归一化仅拟合训练区间，原尺度标签与掩码、互斥目标切分及完整续训复用公共实现。

服务周期采样针对 ResLSTM/CONVGCN 支持窗口间的停运间隙，同时要求各窗口内部及近期到完整目标跨度连续，仍不跨原子文件边界；普通/日历采样保留完整跨度检查。策略写入运行摘要，有间隙数据加入 seasonal_segments_v1 元信息标记，避免旧样本语义无声续训。单空间元素尾批 BN 的运行统计回退提取为独立 runtime_norm 公共模块，网格模型继续使用相同行为，ResLSTM 不依赖整套网格网络。

验证：新增 **31 项通过**，包含四种默认/交换流量及有/无外部网络的修正旧周期轴/SAME 池化/batch 轴参考前向和全部梯度、八种节点/窗口奇偶组合的负无穷尾部补齐独立参考/梯度、完整训练/独立评估/BN/Adam/多步调度器/早停/RNG 精确续训、全部分支真实更新、无跨样本预测及梯度、多步/多目标/单节点/batch=1/float64/输入与 buffer 不变、原始与外部字段推断/未来隔离及历史外部梯度、训练统计/缓存/未来标签变化隔离、真实单样本尾批 BN/CPU AMP、无效结构与协议、两模型服务停运间隙。原先以默认 allclose 判断随机网络微小输出变化的检查改为双精度输入梯度证明，同时保留未使用/未来输入变化后预测严格相同的检查。提取 BN 后与既有网格/清单检查 **73 项通过**（此时尚未加入两项服务间隙用例），最终当前源码完整回归 **642 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。

仓库外实际夹具、训练、独立评估和续训共 **4 条成功命令**，预测/指标逐元素一致，续训第三轮共 24 次参数更新，原始缓存复用，见 [CLI](reslstm-cli-validation.json)。本批明确修正旧错误轴/池化，不声称旧错误路径或论文精度复现。发布配置与较长夹具、参数和输入说明；迁移表更新为 **62/100**，38 条 pending，排除 7 个延期 Common 与 2 个模板后，本期仍有 **29 个交通模型**及最终全量验收待完成。

## commit37：迁移 STSGCN 并修复图掩码与原尺度 Huber 训练

问题：STSGCN 未接入新入口；旧 use_mask 在初始化时只计算一次 mask*adj，保存过期 autograd 图并要求 retain_graph=True；位置嵌入使用原地加法。实际损失调用固定 delta=1 的 Huber，与创建但未使用的 SmoothL1Loss(beta=rho) 不一致；公共损失此前不支持 Huber。模型预设的距离倒数图也不能直接用已有二值/原权重/高斯图替代。

处理：新增独立标准 nn.Module，保留 individual/sharing、GLU/relu、可选时空/首层嵌入、逐预测步独立输出头和默认四块结构。三帧图是只读元信息的独立 buffer，带掩码乘积在每次前向重新创建；位置嵌入不修改输入。共享模式按明确的时间、样本和节点顺序打包窗口；关闭的嵌入不创建无效参数，网络不保留空包装。块内最大聚合要求同宽，历史/配置/图/输入形状在运行前给出明确错误。

新增 data.graph.name=inverse_distance：正距离转倒数、缺边保留 0、显式零距离按旧规则取 1，无权重图仍保持二值；有向与对称规则明确，负距离拒绝，缓存按图选择隔离。旧倒数运算改用带 where 的除法，避免显式零距离先除零。默认 symmetric=true 使用公共 max(A,A.T)，不保留两方向不同距离时的行顺序覆盖行为。

公共 ForecastLoss/配置增加 huber 与 masked_huber、正有限阈值 delta；损失与掩码使用原始目标，保持 float64，AMP 在 float32 上计算。STSGCN 默认保留实际使用的 delta=1 Huber，移除无效模型 rho；无需专用外层训练循环，即可训练、独立评估、完整续训和按有效目标数累积梯度。其他任务的损失枚举保持原约束。

验证：新增 **54 项通过**，包括 16 种旧网络前向/全部梯度与默认四块对照、三帧有向图/静态所有权、连续三次掩码参数更新（不使用 retain_graph）、非原地输入/样本独立、多步/多目标/batch=1/float64、原尺度 Huber 阈值/有效计数/梯度、距离倒数缺边/零距离/方向/缓存与无权重回退、两种模块完整训练/独立评估/mask/Adam/调度器/RNG 精确续训、单节点空空间图尾批 CPU AMP，以及错误参数检查。

仓库外发布夹具、训练、独立评估、续训共 **4 条命令成功**，预测与指标完全一致，续训第三轮共 9 次参数更新，缓存 miss/hit/hit，见 [CLI](stsgcn-cli-validation.json)。当前完整回归 **696 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。迁移表更新为 **63/100**，37 条 pending；排除 7 个延期 Common 和 2 个模板后，本期仍有 **28 个交通预测模型**及最终全量验收待完成，不声明主线全部完成。

## commit38：迁移 STPGCN 并补齐可训练层、真实日历与空间跳数

问题：旧关系推断/图卷积/特征选择保存在普通列表中，六个高斯组件也未注册，导致对应参数不进入常规优化器、state_dict 或设备迁移。范围掩码原地改写空间跳数，位置编码只剩 0/1；不可达索引缺乏独立含义。最终 GLU 写死 C*4，只能适配 L=3。旧数据以样本序号猜测日历，points_per_hour 使用 time_intervals//60，且首次切分还引用了未定义 num_samples；无法可靠对应真实起始星期、午夜和采样频率。

处理：新增独立标准模型、专用轻量 CalendarWindows 适配及预设。所有层与高斯组件注册为 ModuleList，融合宽度按 C*(L+1) 构建；静态有向最短跳数与范围掩码分别存为 buffer，不改动只读图。跳数 0/1/2 等保持真实含义，不可达或超界使用独立 alpha+1 索引；alpha/beta 可为零。保持空间对/转置、GLU、LayerNorm、跳连和逐目标预测布局，不添加模型内部损失、设备常量或日志。

公共窗口仅额外保存原始 clock 引用，STPGCN 按已有窗口索引生成独立的 int64 历史星期/时段；日历不会受归一化、浮点舍入或目标值影响。使用真实 UTC 时间戳，星期一为 0，steps_per_day=86400/time_intervals；保留公共互斥切分、训练统计、尾批与源版本缓存/续训检查。标准 model_inputs 已支持 history_features，沿用原尺度 masked_mae 及公共 train_loop，无需复制外层训练或评估。

旧六个高斯分数是可分离的空间/时间加和，本轮做数学等价分解：先对完整历史聚合空间，再按查询的 beta+1 个因果时段聚合时间。关系权重只存 [N,N] 与 [B,T,beta+1]，不重复构建每个查询的 [B,beta+1,N,N] 大张量；没有改变为 softmax 或近似算法。稠密空间模型本身仍保留，不声称实测大城市加速或论文精度。数值参考先明确纠正上述旧错误，再独立运行原完整关系与逐时刻计算，核对预测及所有梯度。

验证：新增 **36 项通过**，包括六组层数/跳数/时间范围/特征/目标组合与默认 64/3/8 网络的全部梯度对照、关系因子精确对照、SciPy 独立最短路、真实跳数/不可达/参数注册、因果/样本独立/无标签/batch=1/float64、六种采样间隔跨周末午夜、真实原子日历/缓存/互斥目标/训练统计/未来标签隔离、全部关系/卷积/选择层实际更新、完整训练/独立评估/Adam/调度器/RNG 精确续训、单节点 [5,5,5,5,1] 尾批 CPU AMP、spawn 和错误参数检查。

仓库外夹具、训练、独立评估及续训共 **4 条命令成功**，预测与指标完全一致，续训第三轮共 9 次参数更新，24 个高斯参数张量写入检查点，缓存 miss/hit/hit，见 [CLI](stpgcn-cli-validation.json)。当前完整回归 **732 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。迁移表更新为 **64/100**，36 条 pending；排除 7 个延期 Common 和 2 个模板后，本期仍有 **27 个交通预测模型**及最终全量验收待完成。

## commit39：迁移 DGCN 并纠正粗周期轴、图状态与固定长度

问题：旧网络在注意力/掩码/输出中写死 60/48/12，其他周期数或预测长度会切出空片或形状错误；粗注意力 view 把时间位置交错到周期轴。无条件 squeeze 和 float32/device 常量破坏单样本与 double；图残差 Parameter.to(device) 存在迁移后不注册风险。空间注意力的 LayerNorm 从未执行；K=1 却始终创建两个 Chebyshev 项，无法运行。训练 MSE 的零值规则也不能仅凭名字判断。

处理：新增独立标准 DGCN 与 dgcn_cpt 协议，默认保留 2/1/2 个 12 步周期、两层动态图、64 通道、3 阶图卷积和 3 点时间核。模型按只读 CPT 元信息恢复类别及远近顺序，粗注意力显式将整块 H 帧送入通道组；时间掩码、输出头和 LayerNorm 尺寸由 M/H/N 推导。支持单类别/任意非空周期数、多目标、非默认奇数核与头数，保留默认四组输出的计算形式，不用简单网络替换旧动态图。

图使用未归一化 D-A 的最大实特征值缩放，零图/单节点有有限回退，静态矩阵/单位项/掩码注册为 buffer，图残差是直接注册的 Parameter。逐行除法替代对角矩阵乘法，保留原 epsilon 加法并限制接近零的分母；动态 LSTM 状态随输入精度/设备创建。删除未使用的注意力归一化；K=1 不再创建无用图参数。公共 singleton BN 以同一回退逻辑支持 1d/2d，单节点单样本真实尾批可训练，不复制样本。

模型只读取 X；保留原尺度 MSE、Adam 0.0005、指数 epoch 调度 gamma=0.92、关闭早停。旧默认 masked_mse_torch 实际只忽略缺失值，因此明确 mask_value=null，零目标参与训练；评估掩码仍按公共协议独立设置。全部训练、独立评估与 dropout/BN/优化器/调度器/RNG 状态恢复使用公共运行时。

验证：新增 **31 项通过**，含默认 60 帧下 4/8/64 通道的新旧前向/全部有效梯度与训练 BN 状态对照（明确纠正粗周期轴和旧状态精度），粗块逐索引输入、3 种 Chebyshev 阶数/奇数核、未归一化有向拉普拉斯/空图、行归一化抵消、6 组不同周期/预测窗/头数/特征/目标/batch=1/float64、无标签/样本独立/静态状态所有权、真实默认 12 步完整数据、训练统计/目标隔离/缓存、完整训练/独立评估/dropout/BN/Adam/指数调度/RNG 精确续训，以及普通和单块 singleton CPU AMP 尾批。核心与清单检查合计 32 项通过。

仓库外夹具、训练、独立评估与续训共 **4 条命令成功**，预测/指标完全一致，续训第三轮共 15 次参数更新，缓存 miss/hit/hit，见 [CLI](dgcn-cli-validation.json)。当前完整回归 **763 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。迁移表为 **65/100**，35 条 pending；排除延期 Common 与模板后仍有 **26 个模型**，最终验收未完成。

新增公共 CPT 待修项：当前以 3600//time_intervals 推算日/周位移，非整小时约数可能产生不精确的物理周期；同时统一限制 horizon<=每小时点数，会错误拒绝仅日/周输入的较长预测。需要单独按每个激活类别的实际秒数验证/计算位移，并按最短激活位移限制完整预测窗，保证所有输入早于目标；已通过的五分钟/小时配置的样本语义应保持一致。

## commit40：按物理周期修复公共 CPT 位移和长预测检查

问题：原适配以 3600//time_intervals 取整后乘日/周小时数。7 分钟的纯周输入会得到 1344 帧，即 564480 秒（6 天 12 小时 48 分钟），而不是 1440 帧/604800 秒的完整一周；2 小时采样则因每小时点数为 0 被拒绝，即使日周期可以精确表示。原统一 horizon<=每小时点数限制还会错误拒绝仅日/周输入的合法长预测。

处理：新增轻量 cpt_offsets，按每个激活类别的真实 3600/86400/604800 秒及其周期倍数换算帧位移。位移必须被采样间隔精确表示；未启用的类别不施加无关频率限制。每段完整输入都必须在目标前结束，因此 horizon 仅受最短激活位移约束。保持 trend/period/closeness 与每类近到远的既有容器顺序、现有分层字段、缓存边界和目标互斥切分。

对原本准确的整小时约数，数组、max_position 和 temporal_indices 完全相同，不强制重训或丢弃原始缓存。原来可运行但周期不精确的配置会获得修正后的 max_position，已有检查点数据元信息校验会在加载模型/优化器前拒绝沿用错误语义；不修改检查点格式，不绕过现有保护。原始时序缓存不存展开窗口，因此同一来源的缓存仍安全复用。

验证：新增 **24 项通过**，覆盖五种既有频率的旧数组逐项一致、7 分钟仅周/七日倍数、2 小时仅日/日周组合、整日采样仅周、无法精确表示的启用周期、完整窗口到达目标时拒绝、非法采样间隔、真实两小时日周期逐帧相差一天/互斥目标/训练统计，以及真实 7 分钟数据的一周对齐和采样身份变化；与多周期模型和定制训练合计 **41 项通过**。完整回归 **787 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。

修复前已用 commit39 实际训练并保存 7 分钟错误周周期检查点；修复后实际拒绝其续训，随后同一原始缓存上按正确 1440 帧重新训练成功。此前小时级 DGCN 检查点仍可继续到第四轮/20 次更新。新两小时日周期 4 步预测完成训练、独立评估和续训，预测与指标完全一致、缓存 miss/hit/hit。新增 6 条仓库外命令中 5 条成功、1 条按预期拒绝，见 [CPT 记录](cpt-cli-validation.json)。迁移表仍为 65/100，本期实际模型仍剩 26 个，最终全量验收未完成。

## commit41：迁移 HGCN 并补齐区域成员、静态分区和图状态

问题：旧 get_input_cluster 只将每个区域第一个节点加两次，其他成员完全没有贡献；ReLU 自适应图的零度行出现除零。支持图和 transmit 保存在普通属性/列表中，不完整参与设备/精度迁移和检查点；区域图标准差为零产生无效尺度。无条件 squeeze 破坏单节点/单区域/单样本，输出 Linear 要求时间轴最终为 1，导致输入写死在 12 帧。h/h_cluster 和 gate 卷积从未参与前向，nhid 也未用于构建网络。

处理：新增原生节点/区域双层网络，保留四个图池化块、三个学习式传递、跳连及原多目标头。区域特征按只读成员矩阵汇总所有组员一次，静态支持/成员/单位图注册为 buffer，自适应图每次前向生成，零度行使用单位行。末层时间核根据剩余长度推导，默认 [3,2] 要求历史至少 10 帧，可显式缩短核；时序注意力/BN 使用明确轴。删除无效图残差/门内卷积和旧无效 nhid，不持有 scaler/logger/device。

专用数据适配仅按静态图做谱分区，不使用任何流量或验证/测试目标。分区选项位于 data.graph，固定局部随机种子并规范组编号；支持自动 min(10,N)、指定组数和 .geo 顺序显式标签，单节点/零图/每节点一组有确定规则。保留组间权重和的原 Gaussian 规则，标准差为零用零图。实际分区进入只读元信息和安全续训检查；分区设置与原始缓存身份分离，改变组标签仍可复用原始时序缓存。使用现有核心 scikit-learn，不新增依赖。

提取公共 runtime_attention，供 DGCN 与 HGCN 共享双线性时间注意力，DGCN 状态键和数值不变。默认 HGCN 保留 32/256/512 宽度、10 维自适应图、2 阶传播、dropout=0.3、六种图支持规则、Adam 0.001、指数 epoch 调度 gamma=0.92 与 masked_mae。

验证：新增 **47 项通过**，含六种图及默认宽度/训练态共八组修正旧聚合后的完整预测/全部有效梯度/BN 对照、独立旧静态图公式、全成员和梯度守恒、谱分区/私有 RNG/显式标签、常量区域图与零度行、六组历史/核/图阶数/特征/目标/batch=1/float64、输入/图/样本独立、分区设置复用缓存/未来目标变化不影响分区或输入、真实谱分区完整训练/独立评估/dropout/BN/Adam/调度器/RNG 精确续训，以及单节点/单区域/单时间步真实尾批 CPU AMP。与 DGCN 及清单检查合计 79 项；完整回归 **834 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。

仓库外实际夹具、训练、独立评估和续训共 **4 条命令成功**，跨进程分区均为 [0,1,0]，预测/指标完全一致；续训第三轮共 6 次参数更新，缓存 miss/hit/hit，见 [CLI](hgcn-cli-validation.json)。对照明确修复旧成员聚合、单轴及静态 dtype，不声称旧错误训练流程或论文精度复现。迁移表为 **66/100**，34 条 pending；排除延期 Common 与模板后仍有 **25 个交通预测模型**及最终全量验收待完成。

## commit42：迁移 GTS，分离图学习训练与确定性评估

原问题：旧 GTS 的 forward 直接读取 y 并返回预测/辅助输出元组，评估仍随机采图；图学习输入取每个训练窗口第一帧，使模型维度随窗口、padding 和 batch 改变。节点配对依赖两张 N²×N 独热矩阵，并对重复节点投影；大图历史若写入通用元信息还会变成庞大的 Python 数值列表。课程学习、图先验 BCE 和前 200 轮截止逻辑混在模型/旧执行器中，输出默认一维，旧 filter_type 和隐藏 dropout 设置没有实际效果。

处理：新增原生 GTS，保留两层时序 CNN、节点/边嵌入、单随机游走扩散 GRU 编码/解码及原特征顺序。forward 只返回预测 Tensor，不读取标签、不消耗图采样 RNG；验证/独立评估使用 argmax 确定图，并由临时适配器在每次完整评估开始时只生成一次，避免逐 batch 重算或跨训练更新保留旧图。训练辅助前向保留 hard Gumbel 直通梯度；训练 step 显式传入 teacher 目标、按 batches_seen 的 NumPy 课程抽样和稳定图 BCE。正则按显式 epoch 截止，原 class-0 raw logit 的 sigmoid BCE 改用数学等价的 BCEWithLogits，不混成 softmax 类概率。无预测有效值但正则开启时仍支持先验更新。

数据适配只从原始训练区间取连续观测，history_channel 默认 0，history_steps 默认 0（全训练区间），可显式取连续尾段；目标列按训练归一化器转换，其余观测不缩放，拒绝日历/已知未来通道。所选历史需有限、连续且至少 2*kernel_size-1 帧，窗口/预测长度/padding/batch 改变不会改变图特征，验证/测试数值不进入图学习。图选择设置与原始缓存身份分离，起止时间/形状/归一化方式进入运行摘要及续训检查。

新增只读 DatasetMetadata.static_data 承载数值静态数组；模型将图历史、先验与掩码注册为 buffer，检查点元信息仅保存 shape/dtype，避免完整时序额外展开为 Python 列表。既有无静态数组模型不增加字段，原续训检查保持兼容。边线性层在激活前按 sender/receiver 精确分解，保留参数与全部梯度，删除立方规模独热配对矩阵和重复宽投影；仍保留稠密 N² 图与随历史增长的 CNN 展平 fc，规模限制和显式 history_steps 已写入文档。

默认保留 64 单元、1 层 GRU、3 阶扩散、10 点卷积核、100 维嵌入，修正 kernal_size 为 kernel_size，多目标宽度由元信息决定。训练策略移入 training.strategy，保留温度 0.5、课程衰减 2000、前 200 轮图正则；删除无效 filter_type/隐藏 dropout 开关。默认 Adam 0.01、多步 epoch 调度与原尺度 masked_mae（不屏蔽零目标）；单节点/单特征步使用公共单观测 BN 回退。评估确定图与原始训练区间图输入属于明确的有意修订，不声称旧随机评估/窗口特征流程或论文精度等价。

验证：新增 **31 项通过**，四组受控 Gumbel 的旧完整预测/图 logits/BCE/全部梯度/BN 对照包含默认 64/100/3 网络及训练模式，另有三组扩散单元与图梯度、配对分解、解析直通梯度、无标签/RNG/样本独立、仅训练序列/缓存/通道/连续性/未来隔离、只读大数组/检查点元信息、显式课程/稳定 BCE/正则截止及空目标、每次评估只算一张新图。真实原子数据完整训练/独立评估/中断续训跨正则截止后，模型、BN、Adam、调度器、Torch/NumPy RNG 和任务状态逐项完全相等；单节点及单特征步真实尾批 CPU AMP 通过。定向合计 **39 passed**，完整回归 **865 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。

仓库外实际夹具生成、训练、独立评估和续训共 **4 条命令成功**，预测形状 [9,2,3,1]，独立评估预测/指标完全一致；续训第三轮共 9 次参数更新，缓存 miss/hit/hit。三个进程的图特征均为训练前 28 帧、形状 [28,3]；完整权重保存对应 buffer，元信息仅 shape/dtype，独立评估不创建训练产物。见 [GTS CLI](gts-cli-validation.json)。迁移表为 **67/100**，33 条 pending；排除延期 Common 与模板后仍有 **24 个交通预测模型**及最终全量验收待完成。

## commit43：迁移 RGSL，修复图数值与重复计算

原问题：旧 forward 用 if self.train 判断模式，实际拿到绑定方法，且两条分支都随机采图，验证/独立评估不稳定。零向量余弦和空学习图的 max(L)-min(L) 会除零；静态图/多项式未注册 buffer，adj/tilde 属性会保留计算图。静态高阶 Chebyshev 用元素乘法而学习分支用矩阵乘法，cheb_k=1 又多生成一项；小于 16 通道会创建零宽门控层。每个时间步及门/更新分支重复生成学习基底和节点专属参数，末层堆叠最终不使用的全时序隐藏状态，输入/输出宽度默认 1。

处理：新增原生显式/隐式双图混合循环网络，保留原节点专属卷积、两支通道注意力、循环门和多步卷积头。输入/输出宽度取元信息，门控中间维至少 1；静态图基底/单位阵/掩码使用 buffer，不持有 scaler/logger/device 或带梯度的图属性。余弦加入安全归一化，空图伪尺度回退 1，静态零图回退 -I；静态/学习基底统一矩阵递推，支持单项阶数。公共未归一化谱缩放提取到 runtime_graph，DGCN 复用相同计算并保持原状态键。学习基底每个前向生成一次，各层门/更新节点参数准备一次；仅非末层保存完整时间序列。

经 [作者采图代码](https://github.com/alipay/RGSL/blob/main/model/RGSL.py) 核对，保留原扰动分数的线性直通梯度，不自行改成 softmax 梯度或额外取对数。training.strategy.temperature 显式设置其尺度（默认 1），训练专用前向使用 Torch Gumbel RNG，普通 forward 只返回确定性预测 Tensor，不读取标签；每次完整评估只计算一次新图基底，不跨训练更新保留图。默认 Adam 0.003、100 轮上限、早停 patience=15、原尺度 masked_mae 排除零目标，没有新增图 BCE。提供 rgsl.json 与夹具命令；缓存、切分、设备、AMP、累积和续训沿用公共组件。

验证：新增 **28 项通过**，四组受控噪声旧完整预测/全部参数与输入梯度对照覆盖默认 64 单元双层/12 步及高阶、多目标（高阶参考明确修正旧静态递推）。另验证原线性直通梯度/大分数硬值、矩阵多项式、空采图/静态零图/零嵌入/单节点有限梯度、低通道/batch=1/float64、无标签/无 RNG/无跨样本影响、基底与节点参数单次准备、每次评估新图。真实原子数据训练/独立评估与随机采图/Adam/指数调度/RNG 中断恢复逐项完全一致；三种真实尾批 CPU AMP 通过。含 DGCN 和清单检查合计 **60 passed**；完整回归与跨进程命令结果见 [汇总](remaining-validation.json) 和 [RGSL CLI](rgsl-cli-validation.json)。

仓库外夹具、训练、独立评估、续训共 **4 条命令成功**，预测形状 [9,2,3,1]，独立预测与指标完全一致；续训第三轮共 9 次参数更新，缓存 miss/hit/hit，静态基底 [2,3,3] 保存为安全 Tensor 权重，独立评估不创建训练产物。默认二阶计算保留，高阶静态递推、确定性评估、低宽门控和退化图属于明确修订，不声称旧错误流程或论文精度等价。清单为 **68/100** 已迁移，32 条 pending；排除延期 Common 与模板后还有 **23 个交通预测模型**及最终全量验收。

本批完整回归：**893 passed，0 failed，0 errors，0 skipped**；[汇总](remaining-validation.json) 保存本次用例与结果文件位置，既有 GTS 及 DGCN 检查均通过。

## commit44：补齐 CCRNN 所需的 RMSE 聚合与梯度累积

接入 CCRNN 时确认其原目标是 masked_rmse。公共运行时原先只开放 MAE/MSE/Huber 等可按有效数线性聚合的目标，尚无 RMSE 的整组开方机制；若直接添加逐批 sqrt 再走原加权平均，评估和累积梯度都会变成另一个目标。本批先补基础能力，不提前把 CCRNN 标记为已迁移。

处理：新增 RootMeanSquaredLoss，复用原尺度、精确原始掩码和有限值检查，返回平均平方误差和有效数这两个可加统计。默认循环每个微批累计平方误差梯度，到更新边界通过可选 gradient_divisor 按整组 RMSE 链式法则归一化，再执行原裁剪/优化器/调度；可选 aggregate 在评估和训练日志中统一开方。现有损失未声明这两个方法时，继续原有效数加权平均与更新规则。无需保存微批计算图或改变检查点格式，完全空组保持跳步，精确拟合选择有限零次梯度并保留优化器状态步进语义。

验证：新增 **16 项通过**，原尺度双精度/四组掩码与独立 RMSE 梯度、精确拟合/全空统计、四种评估批次划分、带/不带裁剪的 SGD 动量/权重衰减/step 调度与直接整批 RMSE 更新对照、真实累积尾组、双层 dropout/shuffle/Adam/metric 调度/RNG 精确续训和实际 CPU AMP。完整回归 **909 passed，0 failed，0 errors，0 skipped**。详见 [修复说明](root-loss_zh.md) 与 [完整报告](remaining-validation.json)。

本批不新增模型迁移计数，清单仍为 **68/100**；排除延期 Common 和模板后，还有 **23 个模型**及最终全量验收。CCRNN 的原始训练区间 SVD、图预处理和原生训练/评估接入继续实施。

## 持续完成的清单

- [x] 基础交通缓存保存、读取、失效和独立目录。
- [x] CRANN 的统一输入、原始目标、完整训练/评估/续训与无标签推理。
- [x] 公共 CPT 按激活类别的物理周期计算位移与历史窗口合法性，修正非小时整除频率和仅日/周长预测的检查（commit40）。
- [ ] 全部未延期模型迁移、实际验证与迁移 CSV 更新（基线 15/100 已迁移；模板及第 8 节 Common 事项另行标注）。
- [ ] 扩大后的完整回归、发布示例与最终逐项验收；在此之前不声明主线全部完成。
