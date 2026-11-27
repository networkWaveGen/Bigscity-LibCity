# 新运行时：阶段一、二

实施依据和逐项状态分别见 [方案](plan_zh.md) 与 [验收记录](execution_zh.md)。本轮入口采用五区块 JSON；原来的扁平 CLI 参数已被子命令替换。旧模型和 Executor 文件保留到后续迁移清理阶段，未迁移模型不会静默回退到旧运行时。

阶段一、二已通过 **184 项**回归。实际支持范围、依赖与验证级别见 [模型迁移表](models_zh.md)，逐项验收见 [最终报告](acceptance_zh.md)。

## GRU 完整例子

从源码根目录运行，数据、缓存和输出都可放在仓库外：

```powershell
python -m pip install -e '.[dev]'
python tools/make_runtime_fixture.py --directory ../runtime-demo
python run_model.py train --config ../runtime-demo/gru.json --set runtime.exp_id=gru-training
python run_model.py evaluate --config ../runtime-demo/gru.json --checkpoint ../runtime-demo/runs/gru-training/checkpoints/best.pt --set runtime.exp_id=gru-evaluation
python run_model.py train --config ../runtime-demo/gru.json --resume ../runtime-demo/runs/gru-training/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=gru-resumed
```

Python 使用同一入口：

```python
from libcity.pipeline.experiment import run_experiment
result = run_experiment('train', '../runtime-demo/gru.json',
                        overrides={'runtime': {'exp_id': 'python-training'}})
print(result.metrics)
print(result.artifacts['best'])
```

`data.data_dir` 是包含数据集子目录的根目录；`data.cache_dir` 与 `runtime.output_dir` 独立设置。JSON 中的相对路径相对于该 JSON 文件，`--set` 和直接传入 Python 字典中的相对路径相对于当前目录。默认值来自包资源。原子数据的 config.json 只提供数据文件与字段元信息，不能覆盖任务的窗口、切分和评估协议。

## 训练与产物

每次实验写入 `<output_dir>/<exp_id>/`，含 resolved_config.json、runtime_info.json、run.log、metrics.jsonl、checkpoints/best.pt、checkpoints/last.pt 和 eval/ 下的 JSON/CSV 指标及可选预测文件。`evaluation.save_predictions=false` 关闭预测导出。TensorBoard 仅在训练且 `runtime.tensorboard=true` 时创建；独立评估不创建优化器、训练记录器或 TensorBoard。

基础安装不要求 TensorBoard。启用该后端时安装 `pip install -e '.[tensorboard]'`；dev 组为旧执行器回归测试保留该依赖。Trafformer 单独需要 `pip install -e '.[transformer]'`。

`best.pt` 是纯模型 state_dict。`last.pt` 保存最近完整轮次的模型、优化器、调度器、AMP、TrainState、早停、最佳权重、Python/NumPy/Torch 随机状态和训练 DataLoader 随机状态，全部使用 weights_only 安全加载。续训可以延长普通调度器的 epochs；按总更新数定义的 OneCycle/WarmupCosine 必须保持原定 horizon。数据选择、损失、优化器和累积设置不能在续训时悄悄改变。

last.pt 格式版本为 2，还校验只读数据元信息、采样布局和归一化状态。ASTGNN 的两阶段计划同样必须保持一致。旧 `.m/.tar` 和本轮开发中较早的 runtime 格式不自动迁移。换成其他模型时，CLI 的 model.name 选择新预设并丢弃用户 JSON 中前一个模型的结构参数；其余显式实验参数仍按优先级保留。

`training.optimizer.groups` 接受模块前缀、lr、weight_decay 和 freeze。调度器显式设置 interval=step/epoch/metric。梯度累积按真实有效目标数归一化，并在每轮最后一个不足累积组的尾批完成更新；只有实际优化器更新才步进 step 调度器。默认循环支持最小化或最大化指定监控指标、早停、CPU/CUDA/MPS 设备选择和 AMP。当前实际验收硬件是 Windows CPU，未声称执行了 CUDA/MPS。

## 本阶段的数据边界

数据适配继续复用现有滑窗样本、训练集 padding 和训练集 scaler 拟合规则，便于与基线数值对照。验证和测试保持真实样本数。结果中的 `split.semantics=legacy_window_count` 明确表示尚未实施阶段三的目标时间轴互斥切分；已知未来特征与标签的数据语义整理同样留在阶段三。

ASTGNN 使用 data.sampling.name=cpt 及只读采样布局，实际历史长度与请求窗口的差异进入运行摘要。data.graph 控制二值/加权/距离图、对称化和自环，模型预设不会悄悄更改统一切分、窗口或指标协议。

## 无训练权重的地图匹配

```powershell
python tools/make_runtime_fixture.py --directory ../map-demo --task map_matching
python run_model.py evaluate --config ../map-demo/hmmm.json --set runtime.exp_id=map-evaluation
```

HMMM 直接评估，training 区块为空；不需要 checkpoint，不支持 train/resume。这里的轻量原子适配要求双端点 LineString 路段及每条轨迹的真值路线，不读取旧 pickle 缓存。预测保存为 predictions.json；metrics.json/csv 为逐轨迹指标的宏平均，metrics_by_trajectory.json 保留明细。它与交通模型共用实验目录与结构化结果，未强行套用窗口、损失或优化器配置。
