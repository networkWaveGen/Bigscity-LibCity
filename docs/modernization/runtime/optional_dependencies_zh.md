# 旧依赖与模型后续迁移登记

本表依据当前源码与 pyproject.toml 登记替代步骤与已完成结果，不要求运行基础预测模型时安装其他模型的可选依赖。全模型逐项状态在 [model-migration.csv](model-migration.csv)。

| 模型/组件 | 当前源码依赖 | 具体处理步骤与验收门槛 |
| --- | --- | --- |
| GeoSAN | 原 TorchText Field 与 NLTK 已移除，geosan 可选组已删除 | 已用标准库词频表、固定 PAD=0/词频降序/词典序同频规则、字符串滑窗与 pad_sequence 替代。静态 POI 建词表，训练访问统计隔离；修复经纬度和 KNN 弧度、有限候选/访问耗尽回退，保留加权二元目标。已接入原生模型、采样 step 与公共训练，覆盖四种旧有效前向/梯度、采样分布、全词表排名、精确续训；显式阻止 TorchText/NLTK 导入后实际短训通过。具体协议差异见 models_zh.md。 |
| Node2Vec、DeepWalk | embeddings 组的 gensim>=4.4,<5（NumPy 2 兼容） | 已接入独立外部训练回调，保留 Gensim skip-gram 后端；稀疏游走/有界缓存/流式语料，安全 Tensor 向量权重与真实 node_ids。完成固定游走、Gensim 直接数值对照、跨进程确定性及实际 CLI；无需 PyTorch 优化器，独立评估不导入 Gensim。不声明验证/调参/完整续训能力，不保存或加载 Gensim pickle。 |
| GeomGCN | 新运行时已移除 DGL 依赖，旧数学源码作为参考保留 | 原生 PyTorch 边分区消息聚合、完整有向入度归一化与静态 buffer；保留 LibCity 的两分区属性自编码器，不新增论文的几何邻域预处理。修复诱导子图混边、结果未写回及非默认拼接维度。已覆盖独立稠密公式/全部梯度、16 种合并设置、完整训练/评估/精确续训、稀疏图、空图与 CPU AMP；实际阻止 DGL 导入后完成生命周期。 |
| DMVSTNet、GMAN、STAGGCN、STMGAT 等 | 具体可选组合见完整 CSV 的 dependency_extras | 仅在选中时加载。分离图预处理与前向，将静态图注册 buffer；优先对简单消息传递评估用 PyTorch sparse/index_add 重写的可行性，保留边方向、归一化和稀疏度对照。复杂层先记录可用环境后再决定替换。 |
| STWave、PDFormer 等 | fastdtw、PyWavelets、tslearn 等独立组 | 将 DTW/小波/聚类放在明确的数据变换步骤，记录变换配置和产物；可离线准备时不阻塞基础安装。替代算法必须记录近似误差与图结构变化，不能只删除 import。 |
| STGODE、STGNCDE | torchdiffeq 等 | 保持 ode 可选组，单独核对求解器设置、浮点类型、梯度和短训练；不将“不是核心依赖”解释为必须重写求解器。 |
| STTSNet、MultiSPANS 等 | transformer、multispans 等 | 按模型选择加载，逐项验证网络构建与输入布局；额外依赖要求与实际验证分开登记。 |
| 旧 Hyperopt/Ray 管线 | 旧 Executor 与 tuning.py | 新 run_hyper.py 已使用统一运行时和标准库有限搜索。旧内部接口及 tune 依赖别名已在阶段五删除；后端选型仍留在后续讨论，不添加新的调参框架。 |

37 个特殊交通 Dataset 的深度整理、Common 变体合并、多机多卡、LLM 与分发形式继续按方案延后。阶段四按五类任务的代表组合验收，CSV 的 pending 条目明确不是“新运行时已支持”。
