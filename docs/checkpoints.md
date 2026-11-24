# 检查点保存、加载与迁移

公共加载统一经过 `libcity.utils.checkpoint`，默认 `weights_only=True` 并显式传递 `map_location`。检查范围包含交通预测和轨迹执行器、按 epoch 恢复、PDFormer 初始化、GeoSAN 模型加载及调参最佳结果读取；库内其余位置没有绕过该公共入口的 `torch.load`。

| 已有格式 | 处理方式 |
| --- | --- |
| `(model.state_dict(), optimizer.state_dict())` | 保留原保存格式；交通和轨迹公共执行器恢复模型及优化器 |
| `model_state_dict / optimizer_state_dict / epoch` 字典 | 保留原 epoch 格式，并在加载时使用执行器目标设备 |
| `state_dict` 字典别名 | 公共恢复函数识别；可迁移为规范 epoch 字典 |
| 仅模型 `state_dict` | 可用于加载权重；请求恢复优化器时明确报错，避免静默丢失训练状态 |
| 完整模型对象，或包含模型/优化器对象的字典 | 不自动开启对象反序列化；使用显式迁移命令转换为 CPU 状态字典 |

普通状态字典迁移：

```shell
python tools/migrate_checkpoint.py old.m migrated.m
```

只有确认旧文件来源可信时，才对完整对象格式使用：

```shell
python tools/migrate_checkpoint.py old-object.pt migrated.pt --trust-legacy-pickle
```

输出使用独立文件，保留原件。迁移不会自动推断未知自定义封装或替换旧模型类；不存在的旧类仍需由原代码环境先导出状态字典。

集中检查通过真实优化器更新创建非空 Adam 状态，再经交通、轨迹及 epoch 三种实际执行器入口恢复，比较下一步损失、模型参数和 Adam 动量与未中断训练完全一致；同时覆盖上述格式迁移和模型权重单独加载。设备参数转发检查包含指定 CUDA 设备，但实际运行环境为 CPU，不声称已执行真实 GPU 训练。旧格式本来没有保存的调度器与随机数状态不会凭空补齐，因此不承诺任意带随机性的实验恢复后逐位复现。
