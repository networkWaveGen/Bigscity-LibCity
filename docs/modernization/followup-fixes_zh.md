# LibCity 补充问题修复说明

- 日期：2026-11-25
- 仓库：networkWaveGen/Bigscity-LibCity
- 分支：`refactor/libcity-modernization`
- 本轮提交标题：`fix: handle remaining tail-batch and state regressions`

本说明覆盖原有 5 项修复和本次补充的 3 项修复，共 8 项。按照用户最新要求，本次修改合入原 `829be26 — fix: handle remaining tail-batch and state regressions`，通过 amend 替换这一项提交，更早的提交不修改。AgentCity 不在本轮范围内。

开发基线为上一轮提交：[`4a2002c` — fix: correct evaluation accounting and runtime control edge cases](https://github.com/networkWaveGen/Bigscity-LibCity/commit/4a2002cf99b5f0ce2b6daa1f6f4cb107359f46eb)。这是上一轮修复之后的补充修复，之前的说明和验证记录仍保留。

## 完成情况

- [x] CSTN 按实际批大小运行，支持不足一个完整 batch 的尾批。
- [x] 补齐派生执行器的验证损失加权，避免小尾批被赋予过大权重。
- [x] 修复用户 JSON 中 `train: false`、`saved_model: false` 被默认参数覆盖的问题。
- [x] 轨迹预测恢复最佳模型时，同时恢复对应的优化器和调度器状态。
- [x] 同一个交通状态数据集对象可以重复调用 `get_data()`。
- [x] 切换数据列或源文件后不再误用其他配置的缓存。
- [x] ETA 导出完整保留同一用户的多条轨迹。
- [x] 四种归一化方式处理零分母，正反转换保持一致。

本次检查发现的第 4 项日志问题，按用户本次指定范围暂不修改。

## 1. CSTN 使用固定 batch_size，尾批会 reshape 失败

**原来的问题**

CSTN 在 `forward()` 中多次使用配置里的 `batch_size` 做 reshape。配置为 2、实际尾批只有 1 个样本时，张量元素数量不匹配，推理报错。上一轮取消验证集和测试集重复补样后，这个问题会直接影响 CSTN。

**如何修复**

在前向计算开始时读取 `batch['X'].shape[0]`，后续三处涉及 batch 维度的 reshape 全部使用这个实际值。配置中的 batch_size 仍用于组织 DataLoader，不再决定模型内部实际输入的形状。

修改文件：`libcity/model/traffic_od_prediction/CSTN.py`。

**如何验证**

使用真实 CSTN 的精简结构，在配置 batch_size=2 的情况下，分别输入 1、2、3 个样本，检查预测形状、有限损失和反向传播梯度。三个情况全部通过。

## 2. 部分派生执行器仍然直接平均每个 batch 的验证损失

**原来的问题**

上一轮修复了公共执行器，但 DCRNN、MTGNN、PDFormer、ASTGNN、SSTBAN、FOGS 覆写了验证方法，仍然把每个 batch 的平均损失等权平均。

例如三个样本的损失为 1、2、100，batch_size=2。两个 batch 的平均损失为 1.5、100；直接平均得到 50.75，而按样本数量汇总应为 103/3，约 34.333。验证损失用于早停、学习率调整或选取最佳轮次时，会受到这种偏差影响。

**如何修复**

新增 `libcity/utils/training.py` 中的 `SampleWeightedMean`，统一使用：

```text
验证损失 = Σ(当前 batch 的平均损失 × 当前 batch 的真实样本数)
         / Σ(当前 batch 的真实样本数)
```

公共 TrafficStateExecutor 和上述六个派生执行器都改用该工具。轨迹预测执行器的验证损失也使用同样的样本加权方式。前七个执行器遇到空验证集会明确报错。

这一改动汇总的是模型已经返回的 batch 平均损失；没有修改模型自己的损失公式，也没有把带掩码损失重新定义成“按全数据有效元素数归一化”。ChebConv 的整图掩码验证没有改成按 batch 样本加权。

修改文件：

- `libcity/utils/training.py`
- `libcity/executor/traffic_state_executor.py`
- `libcity/executor/dcrnn_executor.py`
- `libcity/executor/mtgnn_executor.py`
- `libcity/executor/pdformer_executor.py`
- `libcity/executor/astgnn_executor.py`
- `libcity/executor/sstban_executor.py`
- `libcity/executor/fogs_executor.py`
- `libcity/executor/traj_loc_pred_executor.py`

**如何验证**

对前七个执行器运行各自真实的验证方法，使用可控模型和 2+1 的不等长批次，确认都返回 103/3；逐个检查空验证集报错。另行检查轨迹预测验证阶段的同类加权结果。

这些测试验证的是执行器汇总逻辑，不等于完成了六个复杂模型的全量训练。

## 3. JSON 设置不训练或不保存，实际仍可能训练、保存

**原来的问题**

ConfigParser 和入口函数预先把 train、saved_model 默认设为 True。加载用户 JSON 时，只补充尚不存在的键，因此 JSON 中的 false 无法覆盖已经写入的 True。只在 pipeline 中读取最终配置，不能解决配置解析阶段已经丢失 false 的问题。

**如何修复**

将入口函数、ConfigParser 和命令行解析中的“调用者未指定”表示为 None；先合并用户配置，再为仍未设置的开关补上 True。

最终优先级为：显式 other_args > 显式函数参数 > 调参配置 > 用户 JSON > 默认配置。未指定的命令行开关不会覆盖 JSON；显式命令行开关仍能覆盖 JSON。对已设置的非布尔值明确报错，地图匹配任务继续强制不训练。

同时调整 HyperTuning 的参数默认值，避免它提前塞入 True。普通实验的 train=false 进入只评估路径；若检查点不存在，沿用上一轮修复的明确报错。调参 objective_function 本身要求训练，因此传入 train=false 时明确拒绝，并提示使用 run_model 做只评估实验。

修改文件：

- `libcity/config/config_parser.py`
- `libcity/pipeline/pipeline.py`
- `run_model.py`
- `run_hyper.py`
- `libcity/executor/hyper_tuning.py`

**如何验证**

使用真实 ConfigParser 检查 JSON、显式参数、other_args 和默认值之间的优先级；运行真实 run_model.py 参数解析入口，分别检查省略开关和显式传入开关的行为；检查 JSON 只评估且无检查点时不会启动数据准备，以及调参入口拒绝禁用训练。

Ray、Hyperopt 等可选后端的完整运行仍属于单独的兼容性工作，本轮没有宣称这些后端已经适配现代版本。

## 4. 轨迹预测恢复了最佳模型，却保留了最后一轮优化器

**原来的问题**

轨迹预测每轮临时保存的只有模型参数。训练结束后恢复最佳轮次的权重，优化器却仍保持最后一轮状态。Adam 的步数和动量可能与模型权重不属于同一个轮次，导出的检查点也会携带这种不一致。

原实现还使用所有实验共用的临时目录，训练后遍历删除目录中的文件。

**如何修复**

当验证指标刷新最佳值时，将该轮的模型、优化器和学习率调度器状态复制到 CPU 内存。训练结束后，成对恢复最佳模型与优化器，并恢复对应调度器状态。

普通训练不再逐轮写入共享临时检查点目录，也不再清扫该目录。load_best_epoch=false 时保留最后一轮状态。没有可恢复轮次或验证准确率不是有限数值时，明确报错。

公开保存格式仍是原来的“模型状态、优化器状态”二元组，因此现有读取流程保持兼容。调度器只在本次训练结束恢复最佳状态时同步恢复；本轮没有扩展磁盘检查点格式来保存调度器、随机数状态或完整断点续训上下文。

同时将轨迹预测 sample 模式验证和评估时的目标标签长度，改成当前预测样本数，避免尾批仍按配置 batch_size 构造标签。

修改文件：`libcity/executor/traj_loc_pred_executor.py`。

**如何验证**

执行两轮真实 Adam 更新，让第一轮验证指标最好，检查结束后的模型权重、优化器步数和调度器状态均回到第一轮；关闭最佳轮恢复时保留第二轮状态。检查保存后的模型/优化器配对，并比较继续一次真实更新的结果。另行检查 sample 模式在 2+1 尾批下，验证和评估的标签数量都与预测数量一致。

测试使用小型模型和人为可控的验证分数，用于定位状态一致性，不代表完整轨迹数据集上的精度验证。

## 5. 同一数据集第二次调用 get_data() 会失败

**原来的问题**

第一次 get_data() 把 self.data 标记成已加载，但没有保存该方法中用于预处理的局部数组。第二次调用跳过加载，后续却继续访问局部空列表的 shape，触发异常。

**如何修复**

如果训练、验证、测试三个 DataLoader 都已经准备好，直接返回原来的三个对象，避免重复加载、归一化和扰动。将“已加载”的标记推迟到整个预处理及 DataLoader 创建成功之后，避免失败时提前留下完成标记。

修改文件：`libcity/data/dataset/traffic_state_datatset.py`。文件名沿用仓库原有拼写。

**如何验证**

分别覆盖从数组生成数据和从 npz 缓存读取数据两种路径；连续调用两次 get_data()，检查三组 DataLoader 的对象身份、数据值及 scaler 参数保持一致，且不会再次生成数据。

## 6. 切换数据列或源文件后误用旧缓存（本次补充）

**原来的问题**

缓存名仅编码部分配置，没有包含 data_col、data_files 等数据选择信息。先读取 speed，再把 data_col 改为 flow 时，两次实验指向同一缓存。复现中 flow 应读取 109，却得到旧 speed 缓存中的 9，程序不报错。

**如何修复**

在 TrafficStateDataset 中集中生成版本为 v2 的缓存标识，内容包含数据集类、已有窗口和切分参数，以及数据列、外部列、数据/地理/关系/外部文件、输出维度、时间间隔和相关预处理配置。对规范化后的配置 JSON 生成摘要，用于命名缓存文件，区分不同配置，并缩短文件名。

TrafficStateDataset 及点、网格、OD 数据集和会重新设置缓存路径的相关子类，共 16 处缓存路径构造统一使用这个方法，避免子类覆盖掉新的标识。旧名称的缓存保留，首次运行新代码时按新路径重新生成。

公共六数组 npz 缓存还保存完整的 cache_config 文本；读取时逐项所对应的完整配置文本必须匹配，缺失或不匹配就重新生成，且先关闭旧文件再重建。读取明确关闭 pickle。自定义格式的数据集沿用自己的读写格式，通过公共路径标识隔离本次涉及的配置。

这里只对**配置字符串**生成摘要，没有对源码或数据文件内容做哈希校验。原始文件在同一路径被原地改写时，仍需要显式刷新对应缓存或设置 cache_dataset=false。道路表征、轨迹及独立辅助缓存没有宣称完成统一改造。

主要修改：`libcity/data/dataset/traffic_state_datatset.py` 及其设置缓存路径的相关子类。

**如何验证**

使用实际 geo/dyna 小数据文件分别切换数据列和源文件，检查结果来自所选列/文件，且相同配置可以直接复用缓存。另行测试缓存配置元数据缺失、被替换时会重新生成；检查旧名称缓存文件仍保留且不会被误读。

## 7. ETA 导出同一用户的多条轨迹时丢失早期记录（本次补充）

**原来的问题**

判断用户是否存在时使用数字 UID，写入字典时却使用字符串 UID。每处理一条轨迹，都可能重新初始化这个用户的结果字典。复现中同一用户输入轨迹 101、102，导出只剩 102。

**如何修复**

读取 UID 后立即转成字符串，存在性检查、用户字典初始化和轨迹写入都使用同一个字符串键。同一用户的不同轨迹继续使用各自的 traj_id 保存。

修改文件：`libcity/executor/eta_executor.py`。

**如何验证**

分别以 batch_size=1 和 2，运行真实 ETAExecutor.evaluate() 和 ETAEvaluator，包含同一用户的两条轨迹及另一个用户的轨迹。检查跨 batch 和同 batch 的轨迹都完整保留，预测值正确，汇总 MAE 不变。

## 8. 常量或全零数据归一化产生 NaN/Inf（本次补充）

**原来的问题**

StandardScaler 在标准差为零、MinMax01/11Scaler 在最大值等于最小值、NormalScaler 在最大值为零时，直接除以零。即使输入都是有限值，转换结果也可能变成 NaN/Inf。

**如何修复**

增加公共零尺度处理：分母恰好为零时使用尺度 1，否则保持原尺度。四种 scaler 的正向和逆向转换使用同一个安全尺度，保留原有 min/max/std 属性用于记录。

例如训练集都是 5，StandardScaler 将 5 映射为 0；之后输入 7，可以映射为 2 并正确还原为 7。MinMax11 的常量训练值映射为 -1，也可以正确逆转换。没有用极小分母强行放大数据。

修改文件：`libcity/utils/normalization.py`。

**如何验证**

覆盖四种 scaler、NumPy/Tensor 两类输入、常量与普通数据；检查有限值、正反转换、非退化情况保持原公式，以及 Tensor 的反向传播梯度。另用真实 TrafficStatePointDataset.get_data() 检查四种归一化方式下训练/验证/测试批次均为有限值，且能正确还原原始值。

## 集中验证结果

原有 `tests/test_followup_regressions.py` 包含 32 项；本次新增 `tests/test_cache_eta_scalers.py` 包含 27 项，并为旧测试夹具补上缓存配置元数据。加上更早的 70 项，最新完整运行结果为 **129 passed，0 failed，0 skipped**，pytest 报告耗时 17.48 秒。

环境：Windows、Python 3.12.13、PyTorch 2.15.0+cpu、NumPy 2.5.5。

运行命令（仓库根目录）：

```powershell
& 'D:\LibCity 2.0\.runtime-envs\libcity312\Scripts\python.exe' -B -m pytest tests -q --maxfail=3 --basetemp 'D:\LibCity 2.0\.runtime-tests\amend-cache-eta-scalers' --junitxml 'D:\LibCity 2.0\.runtime-tests\amend-cache-eta-scalers.xml'
```

最新机器记录：`docs/modernization/audit/cache-eta-scalers-validation.json`。原有 `followup-fixes-validation.json` 仍记录此前的 102 项结果；更早的 39 项、70 项记录同样保留，没有改写成这次的结果。

测试有两条既有 warning：数据集文档字符串中的反斜杠转义警告，以及 RGSL 小尺寸测试夹具创建零元素张量的提示；没有测试失败。

本轮验证使用 CPU、合成数据及实际读取的小型 geo/dyna 文件，没有进行 CUDA、Linux、真实交通数据精度对照或全部模型训练。以上八类已确认的问题已经处理，这份结果不表示整个仓库不存在其他问题。

## 提交方式

原 `829be26` 首次提交时采用追加提交。本次按用户最新要求，将三项补充修复 amend 到它；其父提交仍为 `4a2002c`，更早的提交保持不变。amend 会产生新的提交号，以目标分支最新提交为准。

amend 前保留本地备份分支 `backup/libcity-before-amend-829be26-20261125`。更新远程时使用绑定旧提交完整编号的 `--force-with-lease`，只更新 `refactor/libcity-modernization`：如果远程已不是原 `829be26ff7ce722412918c8fe8e91e4c82b19c1e`，推送将拒绝，不覆盖他人的新增提交。

没有添加 co-author，没有做文件哈希校验。此次缓存标识中的摘要仅来自配置 JSON，与文件内容哈希无关。
