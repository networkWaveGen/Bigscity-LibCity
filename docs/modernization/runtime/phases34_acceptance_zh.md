# 阶段三、四验收报告

实施基线 `41a334d`。按方案阶段三全部要求及阶段四明确的“五类任务各至少一个代表组合、全模型迁移登记、统一调参入口”验收；阶段四是分批迁移，未将原清单所有模型都声明为已迁移。

**最终完整测试：214 passed，0 failed，0 skipped。** 环境为 Windows CPU、Python 3.12.13、PyTorch 2.15.0+cpu、NumPy 2.5.5。机器记录含全部测试名称与依赖版本，见 [phases34-acceptance.json](phases34-acceptance.json)。阶段三首次完整回归 190 项的记录仍独立保留。

## 对照原方案逐项验收

| 阶段/要求 | 当前实现与权威证据 | 结果 |
| --- | --- | --- |
| 3.1 分开历史、已知未来协变量与训练目标；无标签预测；多变量与 Teacher Forcing | traffic_bundle.py 输出 X/future_features/y/y_raw；test_future_inputs_are_explicit_and_never_read_targets_or_future_observations 在移除或污染标签、修改不可提前获取的真实未来观测后验证预测不变；known_future_columns 显式白名单；已有 DCRNN/ASTGNN 教师步仍显式传入归一化 y | 通过 |
| 3.2 按目标时间窗口互斥切分，历史可跨边界；仅训练区间拟合归一化；原尺度掩码；通用交通及 CRANN | timeline.py 与 traffic_bundle.py；test_target_windows_are_disjoint_and_history_can_precede_boundary；test_raw_zero_mask_survives_normalization_rounding；test_crann_all_four_fields_follow_disjoint_target_intervals；显式时间边界测试；验证集污染不改变训练归一化统计 | 通过 |
| 3.3 数据语义单独提交；真实交通数据记录边界、窗口数、配置、种子与短训练指标 | commit1 f9f75b0；[METR-LA 记录](metr-la-phase3.json)，含原始来源、34272×207×2 原数组、前七天 2016×207 数据、12→12 步、seed=17、GRU 一轮、1388/190/393 窗口、跨边界丢弃 22 个；独立权重评估一致 | 通过；这是实数据功能短跑，不是全量精度评测 |
| 4.1 建立逐模型迁移表，登记接入方式、验证/续训能力、依赖与验证级别；先标准后定制；旧依赖方案 | [100 条 CSV](model-migration.csv) 与 test_runtime_inventory.py 逐项对照 task_config.json；13 个运行时入口对应 15 条已迁移记录，85 条 pending；FNN/GWNET 默认循环，STSSL 独立训练及跨轮次精确恢复；[旧依赖替代登记](optional_dependencies_zh.md) | 通过代表批次与清单要求；未完成全目录迁移 |
| 4.2 五类任务各一个代表：交通、变长轨迹及排名、ETA 秒与 JSON、道路嵌入、地图匹配 | traffic/GRU、traj_loc_pred/RNN、eta/DeepTTE、road_representation/LINE、map_matching/HMMM；test_runtime_tasks.py、test_runtime_stssl.py、test_runtime_map_matching.py；各监督新任务训练/独立评估/精确续训；排名和秒制指标独立数学对照 | 通过 |
| 4.3 run_hyper.py 接统一入口；trial 配置、种子和目录隔离；预处理变化重新准备；仅验证选模，最终单独测试 | runtime_tuning.py；test_runtime_tuning.py 的 8 项；四种窗口/归一化组合重新装配；试验期 test loader 迭代禁令；max Recall 选模；随机种子复现；仓库外 CLI；最终 test_evaluations=1 | 通过 |

## 实际命令与依赖隔离

[phases34-cli-validation.json](phases34-cli-validation.json) 保存从仓库外目录实际执行的 **19 条命令**，包括五类任务示例、四类训练模型的独立评估与续训到第三轮、地图匹配直接运行，以及两个 trial 的调参和一次最终测试。所有进程退出码为 0。

该环境通过基础源码安装，未安装 TensorBoard、einops、Ray、Hyperopt、DGL 或 TorchText；示例均能完成。开发测试环境为验证 Trafformer 单独保留 einops，旧回归测试使用的 TensorBoard 也仅在 dev 环境存在。

## 范围与保留项

- 代表模型的具体能力与限制见 [迁移表](models_zh.md)。STSSL 当前要求历史不少于 9 步且预测 1 步；GWNET 当前使用单通道节点目标。
- DeepTTE 以已知计划路线为输入，实际耗时为目标；不能解释成从未来观测还原行程的在线预测。LINE 输出向量及持出边指标，尚未进行下游道路分类认证。HMMM 适配要求双端点路段及路线真值。
- GeoSAN 尚未替换 TorchText Field，其余 pending 模型尚未接入新运行时。完整 CSV 明确区分旧环境导入记录和新运行时实际验证。
- 37 个特殊交通数据集的深度改造、Common 合并、外部调参后端选型、分发、多机多卡、LLM 与阶段五旧实现清理未开展。旧 Hyperopt/Ray 内部接口仍在源码，新 CLI 不调用它们。
- 实际硬件验证仅为 CPU。实数据记录使用 METR-LA 前七天全部 207 节点，未进行完整四个月精度对照，也未执行真实 CUDA/MPS。

实施分批提交见 [执行记录](phases34_execution_zh.md)。本轮不改写历史提交、不添加 co-author、不做文件哈希校验。
