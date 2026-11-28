# CSTN 网格 OD 原生迁移

CSTN 通过公共 OD 数据层、标准 nn.Module 和默认 train_loop 接入统一训练、独立评估及完整续训。使用 `od_window` 协议，不恢复旧 CSTNDataset 或 TrafficStateExecutor。

## 原问题与处理

| 原问题 | 本次处理 |
| --- | --- |
| GCC 在 `[B,N,N]` 上沿 batch 维做 Softmax，同一样本预测受其他样本影响；batch=1 时相关权重全部为 1 | 沿源区域维进行逐列 Softmax，再计算每个目标区域的加权和；独立检查值、梯度和跨样本隔离 |
| 天气 MLP 输出先重复整个 8 维向量再 reshape，空间位置混入不同特征 | 每个特征通过独立空间维广播，在所有网格上保持同值，与作者的 RepeatVector 后置换通道的含义一致 |
| MLP 输入固定 29 列，ConvLSTM 零状态默认 float32 | 外部维数取只读数据元信息；零状态由输入创建，支持 float64 和 AMP |
| 预测读取 y，外部序列覆盖历史及未来实测天气，多步时可引入未知未来观测 | forward 不读标签。只推进明确已知的未来列，未知天气沿用最后观测，不将实测未来天气作为输入 |
| 模型按 CSV 隐含顺序解释网格，起点和终点都可能错位 | 两个 OD 轴分别按 `.geo` 的实际网格位置转为行优先布局，预测后分别恢复原节点顺序 |
| 无需静态路网却加载关系文件，旧代码持有 scaler/device 和固定批次设置 | 显式 `data.graph.name=none`，不读取 `.rel`；归一化与损失留在数据/训练层，批次从输入推导 |

逐列归一化依据[论文的 GCC 公式 5–7](https://guanbinli.com/papers/Contextualized_SpatialTemporal_Network_for_Taxi_Origin-Destination_Demand_Prediction.pdf)。空间广播同时核对了[作者实现](https://github.com/liulingbo918/CSTN/blob/master/models/CSTN.py)。本次保留 LibCity 的双视角二维卷积、单层 ConvLSTM、直接特征点积 GCC 和 ReLU/Tanh 输出，不额外引入作者实现的投影、余弦归一化或其他网络差异，不声称完整论文实现或精度等价。

## 输入与配置

模型要求单个流量观测/目标，`.geo` 通过 `row_id/column_id` 覆盖完整矩形。原子文件可以是 `.gridod`，也可以是具有相同网格元信息的 OD 文件；训练批次仍使用 `[B,Tin,N,N,1]`，预测返回 `[B,H,N,N,1]`，起点和终点身份不会因网格展开丢失。

- `model.params.n_layers` 默认 3，`c_lt` 默认 75，均为正整数。
- 默认 `minmax01`、历史外部特征及时间编码开启；MLP 按实际历史协变量列数构建。全部关闭时不创建 MLP，ConvLSTM 直接接收 32 个空间特征。
- `data.graph.name=none`，不接受对称化和自环开关。原始缓存不把未使用的 `.rel` 当作来源；完整续训仍保留全目录文件版本的保守检查。
- 历史协变量为 `[B,Tin,E]`，已知未来协变量为 `[B,H,K]`。名称及顺序由元信息明确记录，不把天气附加到流量轴。
- 默认训练为原尺度 `masked_mse`、`mask_value=null`，包含真实零流量；Adam 0.001、权重衰减 1e-6、batch 16、100 轮、50 轮早停，无默认调度器。

多步预测时，先用全部历史预测第一个目标，再把预测流量加入历史窗。外部输入随之推进：日历及 `known_future_columns` 声明的列使用当前预测时刻的已知值，其他列维持最后一次历史观测。这是明确的持续值预测策略，并不意味着未来天气已经被观测到。最后一行未来协变量不参与更远一步预测。

数据沿用公共 OD 组件：原始目标时间互斥切分、scaler 仅用原始训练区间拟合、保留 y_raw、真实验证尾批、未展开的缓存及安全来源检查。模型前向不使用 y/y_raw，也不持有 scaler、logger 或固定 device。

## 发布示例与验收

```powershell
python <源码目录>/tools/make_runtime_fixture.py --directory <新目录> --model CSTN
python <源码目录>/run_model.py train --config <新目录>/cstn.json
python <源码目录>/run_model.py evaluate --config <新目录>/cstn.json --checkpoint <新目录>/runs/cstn-demo/checkpoints/best.pt --set runtime.exp_id=evaluated
python <源码目录>/run_model.py train --config <新目录>/cstn.json --resume <新目录>/runs/cstn-demo/checkpoints/last.pt --set runtime.exp_id=resumed --set training.epochs=3
```

[实际 CLI 记录](cstn-cli-validation.json)：四条仓库外命令成功，发布示例预测 `[9,2,4,4,1]`，独立预测和指标完全相同，第三轮累计 9 次参数更新，原始缓存 miss/hit/hit。独立评估没有训练记录、优化器检查点或 TensorBoard 目录。原尺度 MSE 由导出预测独立核对。

新增 27 项专项检查：四组修正旧 GCC/天气广播/状态类型的完整网络前向、所有有效参数与输入/外部特征梯度；独立逐列相关公式和梯度；跨样本零梯度、两条 OD 轴置换、空间广播；历史和已知未来协变量逐步推进；无关坏关系文件隔离、缓存和未来目标/天气修改不影响验证输入及预测；有/无外部特征、单格/单帧/float64/CPU AMP；两种 scaler 的实际训练、独立评估和 Adam/metric 调度/RNG/最佳模型精确恢复；配置和元信息错误。

与 OD、GEML 的检查合计 83 项通过；完整回归 **1072 passed，0 failed，0 errors，0 skipped**，见 [汇总报告](remaining-validation.json)。这是 CPU 功能和数值契约验收，不是论文精度或所有 CUDA/MPS 配置认证。
