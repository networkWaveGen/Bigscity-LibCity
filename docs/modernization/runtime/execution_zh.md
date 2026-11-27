# 阶段一、二实施与验收记录

本轮按 [实施方案](plan_zh.md) 执行阶段一和阶段二。基线为 `6ddc6ee`，开发分支为 `refactor/runtime-phases-1-2`，交付目标分支为 `refactor/libcity-modernization`。阶段三到五及后续讨论事项不在本轮范围。

## 阶段一

- [x] 从指定基线建立实施分支，并在独立环境安装源码及 dev 依赖。
- [x] 基线完整测试通过 131 项；环境与结果见 `baseline/environment.json`。
- [x] 提取可复用原子数据夹具，固定 GRU 权重、输入、目标和零值掩码；保存 `baseline/gru.json` 的前向与损失对照。
- [x] 分层配置、路径解析、只读元数据与 DataBundle；配置/数据专项检查 38 项通过，GRU 标准模块复用基线权重进行无标签前向对照。
- [x] 默认训练、评估、产物、纯权重和完整续训检查点。
- [x] GRU CLI/Python 等价运行及阶段一验收：156 项完整测试通过，记录见 `phase1-validation.json`。

阶段一同时验证了 step/epoch/metric 三种调度时机、OneCycle 与 WarmupCosine 续训、按有效目标数加权的累积梯度和尾批更新、CPU bfloat16 AMP、可选 TensorBoard、min/max 最佳权重与最近续训状态分离。纯评估构造路径被检查为不调用训练上下文；独立进程导入不加载 Ray/Hyperopt/DGL/TorchText/TensorBoard。旧 CLI 的两项参数测试已随子命令接口更新，其余原有回归保留。

## 阶段二

- [ ] STGCN 静态图 buffer 与无样板训练。
- [ ] DCRNN step_fn、课程学习及续训进度。
- [ ] Trafformer 梯度累积与尾批更新。
- [ ] ASTGNN 多阶段训练、标签提取与反归一化。
- [ ] MTGNN 独立训练及单 batch 多次更新。
- [ ] step/epoch/metric 调度、完整状态恢复、指标全局聚合及预测开关。
- [ ] 一个非交通任务的 DataBundle 与无权重/训练能力约束验证。

最终验收前，未打钩项目均不视为完成。本记录随功能提交更新；本轮不做文件哈希校验，提交不添加 co-author。
