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

## 持续完成的清单

- [x] 基础交通缓存保存、读取、失效和独立目录。
- [x] CRANN 的统一输入、原始目标、完整训练/评估/续训与无标签推理。
- [ ] 全部未延期模型迁移、实际验证与迁移 CSV 更新（基线 15/100 已迁移；模板及第 8 节 Common 事项另行标注）。
- [ ] 扩大后的完整回归、发布示例与最终逐项验收；在此之前不声明主线全部完成。
