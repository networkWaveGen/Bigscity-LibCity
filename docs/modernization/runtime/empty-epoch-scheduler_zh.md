# 无参数更新轮次的学习率修复

问题：当整轮训练没有有效监督、没有可构造的排序负样本，或 AMP 跳过全部参数更新时，optimizer_steps 不增加，但 complete_epoch 仍调用 epoch 调度器。这会在首次 optimizer.step 之前衰减学习率，并触发 PyTorch 的调度顺序警告。

修复：默认 train_loop、MTGNN 子图训练和 STSSL 独立训练都在每轮开始记录 optimizer_steps，并显式传给 complete_epoch。该轮至少发生一次实际参数更新才推进 epoch 调度器。step 调度仍只随实际更新推进；metric 调度继续按每次验证结果调用。验证、早停、日志和检查点仍正常完成，零更新轮次如实记录为零更新。不增加持久状态字段，不修改检查点格式。

验证：公共引擎、定制训练和 STSSL 三个测试文件共 **31 项通过，零失败、零错误、零跳过**。新增用例连续跳过前两轮所有更新，确认学习率和 epoch 调度器都不前进；从第二轮 last.pt 恢复后完成首次有效训练，与连续训练的参数、训练状态和调度器完全一致，且只衰减一次学习率、没有调度顺序警告。原有三类调度器、课程学习、多阶段和 STSSL 动态状态的恢复检查保持通过。

复验命令（临时目录可自行指定）：

```powershell
python -m pytest tests/test_runtime_engine.py tests/test_runtime_custom_training.py tests/test_runtime_stssl.py -q
```
