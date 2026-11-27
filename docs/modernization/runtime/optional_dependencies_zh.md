# 旧依赖与模型后续迁移登记

本表依据当前源码与 pyproject.toml 登记替代步骤，不声称已替换这些依赖，也不要求运行基础预测模型时安装它们。全模型逐项状态在 [model-migration.csv](model-migration.csv)。

| 模型/组件 | 当前源码依赖 | 具体处理步骤与验收门槛 |
| --- | --- | --- |
| GeoSAN | dataset_subclass/geosan_dataset.py 第 8 行导入 torchtext.data.Field，302–309 行使用 Field 分词与 build_vocab；geosan 组仍固定 torchtext==0.6.0 | 用普通 Python 词频表、固定 token→ID 映射和 pad_sequence 替代 Field。先冻结 QUADKEY 的分词、min_freq、词表排序和 padding 输出，再比对层级地理编码、负采样分布、候选排名及独立 train 的损失；元信息保存普通列表，权重与任务状态用安全 checkpoint。完成前不注册到新运行时，不把“安装旧 TorchText”当作现代化完成。 |
| Node2Vec、DeepWalk | embeddings 组的 gensim | 先隔离随机游走与词向量训练，明确不可用梯度优化器、是否支持验证/续训；通过直接训练回调输出统一向量与 node_ids，不调用旧全局缓存路径。对照固定游走、词表和相似度；若替换训练器，单列数值语义变更。 |
| DMVSTNet、GMAN、STAGGCN、STMGAT、GeomGCN 等 | graph/DGL；具体组合见完整 CSV 的 dependency_extras | 仅在选中时加载。分离图预处理与前向，将静态图注册 buffer；优先对简单消息传递评估用 PyTorch sparse/index_add 重写的可行性，保留边方向、归一化和稀疏度对照。复杂层先记录可用环境后再决定替换。 |
| STWave、PDFormer 等 | fastdtw、PyWavelets、tslearn 等独立组 | 将 DTW/小波/聚类放在明确的数据变换步骤，记录变换配置和产物；可离线准备时不阻塞基础安装。替代算法必须记录近似误差与图结构变化，不能只删除 import。 |
| STGODE、STGNCDE | torchdiffeq 等 | 保持 ode 可选组，单独核对求解器设置、浮点类型、梯度和短训练；不将“不是核心依赖”解释为必须重写求解器。 |
| STTSNet、MultiSPANS 等 | transformer、multispans 等 | 按模型选择加载，逐项验证网络构建与输入布局；额外依赖要求与实际验证分开登记。 |
| 旧 Hyperopt/Ray 管线 | 旧 Executor 与 tuning.py | 新 run_hyper.py 已使用统一运行时和标准库有限搜索。旧内部接口及 tune 依赖别名已在阶段五删除；后端选型仍留在后续讨论，不添加新的调参框架。 |

37 个特殊交通 Dataset 的深度整理、Common 变体合并、多机多卡、LLM 与分发形式继续按方案延后。阶段四按五类任务的代表组合验收，CSV 的 pending 条目明确不是“新运行时已支持”。
