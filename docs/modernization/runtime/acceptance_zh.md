# 阶段一、二最终验收

实施基线：6ddc6ee；独立环境：Python 3.12.13 / PyTorch 2.15.0+cpu / NumPy 2.5.5 / Windows。先安装源码及 dev 依赖并通过原始 131 项，再记录固定 GRU 输入、目标、掩码、权重和数值输出。阶段二仅为 Trafformer 追加 transformer 组的 einops，没有安装调参、DGL 或 TorchText 栈。

最终完整回归：**184 passed，0 failed，0 skipped**。机器结果、依赖版本及逐测试名称见 [acceptance.json](acceptance.json)。阶段一 156 项记录与原始 131 项基线分别保留。

文档生成器和 CLI 的训练、评估、续训、地图匹配命令另行实际执行，记录见 [examples-validation.json](examples-validation.json)。基础安装的可选依赖隔离验收见 [minimal-environment.json](minimal-environment.json)。

## 阶段一逐项核对

| 方案要求 | 证据 | 结论 |
| --- | --- | --- |
| 从指定提交建分支；独立安装与原测试基线 | 开发分支 refactor/runtime-phases-1-2；baseline/environment.json 记录安装、版本、131 项结果 | 通过 |
| 可复用小型原子数据夹具与固定数值对照 | tests/runtime_fixtures.py；baseline/gru.json；test_adapter_matches_atomic_baseline_and_metadata_is_readonly | 通过，固定权重下前向/损失一致 |
| 分层配置、实验入口、训练工具、数据适配及产物 | config/runtime.py、data/bundle.py、executor/components.py/train.py/evaluate.py、pipeline/experiment.py | 通过，五区块 JSON、只读元信息、显式路径与 Tensor 字典 |
| GRU CPU 训练/保存/独立评估/续训；Python 与 CLI 一致 | test_gru_cli_python_train_evaluate_and_resume，实际子进程从仓库外目录执行新 CLI | 通过，指标和预测一致，best.pt/last.pt 分离 |
| 纯权重评估不建优化器和训练记录器；缺项、合并、缺失文件、可选依赖隔离 | test_runtime_config_data.py；test_checkpoint_errors_and_optional_import_isolation；纯评估训练构造 spy | 通过；模型切换不会残留旧结构字段；新入口导入不加载 Ray/Hyperopt/DGL/TorchText/TensorBoard |

## 阶段二逐项核对

| 方案要求 | 证据 | 结论 |
| --- | --- | --- |
| STGCN 图元数据与 buffer，零训练样板复用 | runtime_stgcn.py 的 buffer；图模型相同旧权重计算/两次 Adam 更新；统一实验入口测试 | 通过 |
| DCRNN step_fn 读取 batches_seen，Trafformer 累积和尾批 | test_dcrnn_curriculum_resume_retains_progress_rng_and_next_updates；test_trafformer_unequal_horizon_flushes_short_accumulation_group | 通过，保留课程进度并完成最后不足累积组的更新 |
| ASTGNN 两阶段与标签/原尺度，MTGNN 独立单批多次更新；核对参数/LR/更新次数 | test_runtime_custom_training.py 的 9 项测试；旧架构参考及修订训练语义见执行记录 | 通过，教师/微调均可恢复；子图更新及计数可恢复 |
| 三种调度时机、完整续训、尾批/缺失值/多预测步指标与导出开关 | test_runtime_engine.py：5 种调度配置恢复、有效目标数加权、全局统计、min/max、早停、clip、CPU AMP、dropout/shuffle；所有神经模型共用 evaluate_loop | 通过；last.pt 还保存/校验数据元信息与归一化状态 |
| 一个非交通任务，验证容器、结构化结果与能力限制 | test_runtime_map_matching.py 的 5 项；真实 HMMM，train/val=None，无 checkpoint/优化器/writer | 通过；两条真值轨迹 RMF=0、AN=1、AL=1 |

测试按公共行为与代表性覆写分层：公共循环/调度/导出在公共入口验收，定制 train、CPT 数据和直接评估分别补专项。没有把六个模型中的每种参数组合重复列为已认证。

## 产物与运行说明

训练、续训、独立评估和地图匹配命令见 [使用说明](README_zh.md)，支持与验证级别见 [模型表](models_zh.md)。每次运行的 resolved_config.json、runtime_info.json、run.log、metrics.jsonl、checkpoints/ 和 eval/ 均集中在独立实验目录。纯评估没有训练类产物。预测关闭时不残留上次的预测文件。

best.pt 为纯模型权重；last.pt 为版本 2 的安全完整状态。恢复覆盖参数、Adam 状态、LR、epoch/batches_seen/optimizer_steps、早停、最佳选择、Python/NumPy/Torch 随机状态、训练加载器随机状态及自定义任务进度。已经早停的恢复不会额外更新，并仍保留 last.pt。

## 保留的阶段边界

- 阶段三的目标时间轴互斥切分、已知未来特征与观测/标签分离、原始尺度目标的精确掩码，以及真实 METR_LA 短训练记录仍未执行。
- 其余任务/模型、37 个特殊交通数据集、Common 变体合并、调参后端、旧文件清理、分发方式、分布式及 LLM 集成仍按方案后续推进。
- 本轮是 CPU 与可复现小型原子数据验收，没有真实城市数据精度对照，也没有真实 CUDA/MPS 运行。
- 现有文档字符串、旧 DCRNN 稀疏构造与 RGSL 夹具有提示；安全加载稀疏张量的校验提示及显式更改指标的协议提示属于预期输出，测试没有失败或跳过。

以上结论对应阶段一、二的原始要求，没有把后续阶段工作算作完成。本轮共六个新提交，不改写之前的提交，不添加 co-author，不做文件哈希校验。
