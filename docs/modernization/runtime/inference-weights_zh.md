# 独立评估权重的输入一致性检查

## 问题

原先独立评估只检查 `best.pt` 是纯 Tensor 权重、参数名和形状可加载。形状相同并不保证输入含义一致：例如把训练时的 MinMax01Scaler 换成 StandardScaler，或者改变节点、网格位置及已知未来列的顺序，仍可能成功加载。旧映射 buffer 还会覆盖模型根据当前元信息构建的新映射，产生无报错但含义错误的预测。

已在 CSTN 数据上复现旧加载器接受不同 scaler；本次在神经模型公共入口修复，不只对 CSTN 特判。

## 修复

- 创建训练上下文时生成 `inference_spec`：模型名称和参数、影响输入的数据配置、只读数据身份/布局，以及 scaler 的类型和拟合统计。
- 保存最佳权重时，把这份安全字典放入标准 state_dict 的 `_metadata['']['libcity_input_spec']`。state_dict 的键和值仍完全对应模型权重/Buffer，所有值仍是 Tensor；原有 PyTorch 模块版本元信息保留，不混入优化器、训练计数或随机状态。
- CLI/Python 统一独立评估入口重新生成当前输入说明，在 `model.load_state_dict` 之前逐项比较。布局、身份、归一化或模型配置不同会明确拒绝，尚未覆盖模型 buffer，也未进行预测。
- 使用 `weights_only=True`；文件内只使用 PyTorch 支持的普通容器和安全值，不增加自定义反序列化类型。普通 `nn.Module.load_state_dict(torch.load(..., weights_only=True))` 仍可加载该 state_dict。

`best.pt` 仍是可单文件复制的标准权重，不需要旁边的 JSON 文件，也不需要初始化优化器或训练记录器。数据/缓存/输出目录、缓存开关、训练补齐以及训练/评估批次大小不参与输入说明比较。可以更换批次和存储目录，并更改评估指标；数据声明、训练区间统计与输入布局必须相符。

此检查针对输入契约，不对原始文件内容做哈希，也不把原子文件修改时间作为独立评估门槛。只更新测试区间观测、保持训练统计和输入定义不变时仍可评估。完整续训另有原模型/训练设置、数据状态及全部原子文件版本检查，继续保留。Gensim 等外部算法沿用自身的身份与状态校验。

## 现有权重如何处理

新训练生成的 `best.pt` 自动携带说明。缺少说明的旧 `best.pt` 在公共独立评估入口明确报错，不猜测原来的 scaler 或输入身份。

若保留了对应格式 3 的完整 `last.pt`，使用原始配置和原始数据执行一次正常 `--resume`，恢复流程会先验证原来的模型参数、训练/数据配置、数据统计及文件版本，再为其中的最佳权重补入说明。若检查点已经完成所需 epoch，可不增加任何参数更新，重新导出到新的实验目录：

```powershell
python run_model.py train --config <原配置.json> --resume <原last.pt> --set runtime.exp_id=reexport
python run_model.py evaluate --config <原配置.json> --checkpoint <output_dir>/reexport/checkpoints/best.pt --set runtime.exp_id=reevaluate
```

恢复时若已有说明但内容冲突，同样在恢复模型、优化器和 RNG 前拒绝；只为经过完整校验、缺少说明的旧最佳权重补录。原始 `best.pt/last.pt` 不会被改写。

## 验证

新增 13 项：安全 Tensor 键/标准模块版本/独立快照、标准 PyTorch 加载与单文件复制；七种相同形状的输入不一致在参数加载前拒绝；公共评估允许批次/训练补齐变化且不构建优化器/记录器；缺少说明拒绝；旧完整状态零额外更新导出、冲突说明在恢复前拒绝；测试观测变化在输入定义不变时允许。

结合 CSTN、GEML、训练引擎及旧入口检查共 109 项通过。[实际 CLI](inference-weights-cli-validation.json) 覆盖发布 CSTN 训练、单独复制权重的独立评估、不兼容 scaler 明确失败、完整续训；五条命令均符合预期，正确路径预测和指标完全一致。完整回归 **1085 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。
