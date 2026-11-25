# LibCity 运行问题修复说明

日期：2026 年 11 月 25 日。开发分支：`refactor/libcity-modernization`。

本轮以 `fcaf269fe8ca3f020bd7f0f28c8a410fdf136cff` 为父版本，将下述六类问题的实现、回归检查及本说明合为一个新增 commit。既有提交不修改，不执行 amend、rebase 或 force push。

## 1. 验证、测试补齐样本被计入指标

**原问题。** 通用 DataLoader 将训练、验证、测试三个集合都用最后一个样本补齐。评估器把补出的样本当成真实样本，导致指标偏差。三个真实目标为 `1、2、100`、预测均为零、批大小为 2 时，原实现将最后的 100 重复一次，MAE 从应有的 `103/3 ≈ 34.3333` 变为 `203/4 = 50.75`。

**修复。**

- `pad_with_last_sample` 只对训练集生效，保留原训练补齐能力。
- 验证和测试均保留真实样本及不齐尾批，不重复、不丢弃末尾样本。
- 验证集不再打乱顺序。
- 公共交通执行器按实际批次样本数加权汇总验证损失，避免短尾批与整批被赋予相同权重。
- 更新原有尾批检查，使其分别检查“训练可补齐”和“验证、测试不补齐”。

**结果。** 上述例子的 MAE 恢复为约 34.3333，预测归档中也只包含三个真实样本。回归检查同时核对验证损失、最终指标和归档样本数。

**范围。** 验证/测试的批次长度可以小于配置的 batch size。本轮同步修正 STSSL 时间异质性分支的固定批大小标签。按样本数加权是公共标量损失的汇总策略，不等同于对所有自定义 masked loss 重新定义全数据有效元素权重；模型专用损失公式保持原样。

主要实现：[数据加载](https://github.com/networkWaveGen/Bigscity-LibCity/blob/refactor/libcity-modernization/libcity/data/utils.py)、[公共交通执行器](https://github.com/networkWaveGen/Bigscity-LibCity/blob/refactor/libcity-modernization/libcity/executor/traffic_state_executor.py)。

## 2. 关闭模型保存后仍加载磁盘最佳模型

**原问题。** `saved_model=false` 与默认 `load_best_epoch=true` 同时使用时，训练过程没有保存 epoch 文件，结束后却尝试读取，出现 `Weights at epoch 0 not found`。

**修复。**

- 需要恢复最佳模型时，即使关闭磁盘保存，也在验证损失改善时保存最佳状态。
- 关闭磁盘保存时，只保留一份独立的 CPU 内存快照，包含模型和优化器状态；保留状态字典元数据，避免快照共享训练中的 Tensor。
- 训练结束从内存恢复最佳模型和真实优化器状态；开启磁盘保存时沿用原 epoch 文件方式。
- 同步使用该公共机制的 DCRNN、MTGNN、PDFormer、STSSL、TESTAM、TimeMixer、ChebConv 执行器训练分支。

**结果。** 不写模型检查点也能选择并恢复最佳 epoch。检查使用真实 Adam 更新，在两轮训练中让第一轮验证表现更好，确认最后恢复的是第一轮模型参数和非空优化器状态，模型缓存目录没有新文件。

`saved_model=false` 控制模型文件保存；实验日志及 TensorBoard 记录仍正常生成。内存最佳状态会占用 CPU 内存。

主要实现：[检查点工具](https://github.com/networkWaveGen/Bigscity-LibCity/blob/refactor/libcity-modernization/libcity/utils/checkpoint.py)、[公共交通执行器](https://github.com/networkWaveGen/Bigscity-LibCity/blob/refactor/libcity-modernization/libcity/executor/traffic_state_executor.py)。

## 3. 部分模型忽略配置而强制使用 CUDA

**原问题。** STNN、DSAN 已有设备修复，但 ACFM、ACFMCommon、DSTAGNN 等其他实现仍调用 `.cuda()`。CPU 配置下创建 DSTAGNN 曾报 `Torch not compiled with CUDA enabled`。

**修复。**

| 模型 | 本轮处理 |
| --- | --- |
| ACFM、ACFMCommon | ConvLSTM/GRU 初始状态使用输入张量的设备及 dtype |
| DSTAGNN | 位置索引使用输入设备；邻接张量按配置创建，并登记为非持久 buffer |
| RGSL | 图掩码和单位阵使用相关计算张量的设备及 dtype |
| SSTBAN | 时间嵌入、注意力掩码、缺失掩码跟随对应输入/空间嵌入设备 |
| STSSL | 图登记为按配置放置的非持久 buffer；时间异质性分支按实际批次长度和 logits 设备创建标签与打乱索引 |

非持久 buffer 会跟随模型的设备/类型转换，但不会向旧状态字典增加必需键，避免因此破坏历史权重加载。

专项检查还发现 ACFM、ACFMCommon 的门控偏置只分配内存而未初始化，可能引入非有限梯度。本轮将该偏置明确初始化为零，并保留原权重初始化方式。

**验证。**

- ACFM/ACFMCommon 的 LSTM、GRU 子模块在 CPU float64 下完成前向与反向，检查 dtype、设备和有限梯度。
- DSTAGNN、RGSL 完成 CPU 模型预测、损失及反向检查。
- SSTBAN 检查 CPU 时间嵌入与带掩码注意力。
- STSSL 检查短尾批时间损失及图 buffer 的设备/类型迁移。
- 检查模型目录，未再发现本轮扫描模式下的硬编码 `.cuda()`、CUDA Tensor 类型和全局默认张量类型设置。

真实 GPU、多 GPU 和所有注册模型的完整训练没有在本轮执行；上述扫描不代表所有设备相关问题都已被证明不存在。

## 4. 数据量小时测试集错误地包含全部样本

**原问题。** `num_test=0` 时，`x[-num_test:]` 等于 `x[0:]`。例如两个窗口样本按默认比例划分时，曾得到训练集 `[0]`、验证集 `[1]`，测试集却为 `[0,1]`。

**修复。**

- 检查训练/验证比例必须有限且为正，二者之和小于 1。
- 检查实际计算出的训练、验证、测试样本数均大于零；不足时给出明确错误，提示增加数据、缩短窗口或调整比例。
- 测试集使用正向起点 `num_train + num_val` 切片，消除负零切片隐患。
- 非法划分在写入新缓存之前失败。

**结果。** 两窗口边界用例明确拒绝；正常划分检查确认三个集合按时间顺序、互不重叠。

这里的“互不重叠”指生成后的窗口样本分配；没有重写原来的滑动窗口生成算法。已有历史缓存和历史指标不会被自动修改，复核旧实验时应重新生成其预处理缓存。

主要实现：[交通数据集](https://github.com/networkWaveGen/Bigscity-LibCity/blob/refactor/libcity-modernization/libcity/data/dataset/traffic_state_datatset.py)。

## 5. 调参优先级与配置说明不一致

**原问题。** 配置解析先合并命令行/外部参数，再让搜索参数覆盖它，与原注释声明相反。

**修复。** 在 ConfigParser 内先合并搜索参数，再合并显式外部参数。当前约定为：

```text
命令行 / other_args > hyper_config_dict > 用户 JSON > 默认配置
```

**结果。** 外部学习率 0.1 与搜索值 0.001 同时存在时采用 0.1；没有显式外部值时采用搜索值。检查覆盖两种情况。

该优先级适用于 ConfigParser 的参数合并。Ray Tune 的旧后端 API 迁移仍按原约定暂缓，没有在本轮改造成新调参框架。

主要实现：[配置解析](https://github.com/networkWaveGen/Bigscity-LibCity/blob/refactor/libcity-modernization/libcity/config/config_parser.py)。

## 6. 只评估模式在检查点缺失时意外训练

**原问题。** 入口使用 `train or not exists(checkpoint)` 决定是否训练；即使 `train=false`，缺少文件仍进入训练，而且分支使用原函数参数，可能与最终配置不同。

**修复。**

- 训练和保存分支使用合并后的配置值。
- 对需要检查点的任务，只评估且文件缺失时，在创建数据集和启动训练之前抛出明确的 FileNotFoundError。
- 文件存在时加载后评估，不调用训练。
- 传统地图匹配不要求神经网络检查点，继续直接执行评估。

**结果。** 检查覆盖缺少模型、已有模型、显式训练和传统地图匹配四条路径；还故意让原函数参数与最终配置不同，确认按最终配置运行。

行为变化：`train=false` 不再自动回退为训练。复用模型需要选择正确的实验编号和已有检查点；需要训练时显式开启。

主要实现：[公共管线](https://github.com/networkWaveGen/Bigscity-LibCity/blob/refactor/libcity-modernization/libcity/pipeline/pipeline.py)。

## 集中验证

在主体修改完成后，从仓库根目录集中运行完整 tests 目录：

```shell
python -B -m pytest tests -q --maxfail=2 --basetemp ../.runtime-tests/runtime-fixes --junitxml ../.runtime-tests/runtime-fixes.xml
```

实际命令使用上述工作区临时目录的绝对路径。临时目录专供 pytest 使用。

结果：**70 项通过，0 失败，0 跳过，用时 17.15 秒**，其中本轮新增 26 项专项检查。

| 检查模块 | 项数 |
| --- | ---: |
| 检查点恢复 | 10 |
| 各任务组件入口 | 10 |
| 现代核心 | 24 |
| 本轮运行问题回归 | 26 |

环境为 Windows、Python 3.12、CPU PyTorch。逐项名称及环境记录见 [runtime-fixes-validation.json](https://github.com/networkWaveGen/Bigscity-LibCity/blob/refactor/libcity-modernization/docs/modernization/audit/runtime-fixes-validation.json)，新增检查代码见[专项回归检查](https://github.com/networkWaveGen/Bigscity-LibCity/blob/refactor/libcity-modernization/tests/test_runtime_regressions.py)。

运行有两条 warning：原有文档字符串的转义提示，以及 RGSL 小规模检查中零元素层初始化提示。对应检查均通过。

本轮没有更新旧的入口审计 CSV、历史 39 项验收 JSON，也没有进行真实交通数据精度回归、干净 Linux 安装或真实 CUDA 运行。新结果单独归档，不将历史证据覆盖成当前结果。

## 文件与提交范围

- 实现：通用数据加载、交通数据划分、ConfigParser、公共管线、公共检查点/交通执行器、七类派生执行器的最佳模型分支及六个模型文件。
- 检查：新增专项回归文件，调整旧尾批检查以体现新的验证/测试语义。
- 文档：本文件及本轮机器验证记录；现代化 README 增加入口链接。
- 不新增基础依赖，不修改 AgentCity，不执行文件哈希校验。
