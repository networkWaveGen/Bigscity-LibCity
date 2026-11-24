# 各任务入口与注册映射验收

验收覆盖显式实现路径、任务配置与公共组件工厂。Python 调用方直接从实现模块导入，任务配置继续使用注册名称。

```shell
python docs/modernization/audit/entrypoint_audit.py
python docs/modernization/audit/compatibility_report.py --runtime-report docs/modernization/audit/registration_imports.csv
python -m pytest tests/test_component_entrypoints.py -q
```

审计输出写入 [`docs/modernization/audit/`](audit/README.md)。新版脚本检查组件的显式实现路径，输出 `implementation_imports.csv`。`registration_imports.csv` 逐项读取默认配置，解析 RNN/GRU/LSTM 别名，并通过运行时注册表检查模型、数据集、执行器、评估器及编码器。`script_imports.csv` 检查根目录及 `test/` 下脚本的 LibCity 导入。基础训练和可视化 CLI 另做 `--help` 检查。

当前归档的是显式导入改造之前的结果。原验收中 200 个旧公开组件及 84 项任务注册通过导入，其余 20 个组件和 16 项注册缺少可选依赖。新版审计脚本本轮尚未运行，`public_imports.csv` 保留原来的包级导入检查记录。

核查发现上游 `TemplateTLP` 在允许列表中，却缺少任务映射。本轮补齐与其模板输入相匹配的标准轨迹数据集、执行器、评估器和编码器映射。`TemplateTLP` 与 `TemplateTSP` 仍是开发模板，未实现模型算法，不能作为可训练模型宣传。

代表任务的实际工厂检查覆盖交通预测 GRU、轨迹 GRU、ETA DeepTTE、地图匹配 HMMM、路网表示 LINE，创建真实模型、执行器与评估器；另创建真实轨迹与 ETA 数据集及编码器。轨迹 GRU、DeepTTE、LINE 新增损失与反向传播检查，交通 GRU/STGCN 与公共管线训练沿用核心检查。传统地图匹配只检查实际工厂创建，不要求反向传播。

TorchText、DGL、Ray 等旧可选栈的运行兼容性仍待验证。
