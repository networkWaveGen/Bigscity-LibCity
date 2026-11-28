# GEML 原生 OD 迁移

GEML 已接入统一 `run_model.py train/evaluate`、Python API 和公共检查点生命周期；CSTN 仍待迁移。本批使用前一批交付的 [OD 数据组件](od-data_zh.md)，不恢复旧 GEMLExecutor。

## 原问题与处理

| 原问题 | 本次处理 |
| --- | --- |
| SLSTM 把 `[P,B*N,E]` 的细胞状态覆盖成 `[B*N,E]`，下次循环按周期索引取到样本行；隐藏状态原位更新也会破坏反向传播 | 每个周期分别保存隐藏/细胞状态，按 `t % P` 非原位更新；初始状态由输入创建，批次和节点互不串扰 |
| 地理图按固定 batch 和输入窗重复保存，孤立行除零 | 仅注册一个 `[N,N]` buffer，在批次和时间上广播；保留平方距离按行归一化、对角置 1 的原公式，稳定处理大距离和零行 |
| 语义图依据归一化流量的正负号生成，零流量和低于均值的真实正流量语义改变；可微图挂在模型对象上 | 使用数据层提供的原始正值布尔掩码，保留对称连通及度数公式；语义图是单次前向的局部张量 |
| 模型保存 scaler/device，forward 返回预测与辅助量的混合结果 | 标准 nn.Module；forward 只返回预测 Tensor，forward_with_aux 显式提供训练辅助头；反归一化和损失位于训练 step |
| 对归一化后的真实 OD 先求和、再反归一化，仿射变换的偏移只恢复一次 | 在原始目标上求起点/终点边际和；辅助头在外部独立反归一化，再计算原尺度误差 |
| 固定形状、批次边界和旧执行器不能直接复用新运行时 | 显式检查单流量布局、从实际 batch 推导形状，复用公共 train_loop、优化器、调度器、评估与安全完整续训 |

图卷积仍是原来的两层无激活图映射，周期记忆仍保留原 softmax 门和存储 `tanh(cell)` 的约定。没有改成标准 sigmoid LSTM。预测窗内固定使用最初历史对应的语义图，与旧自回归策略一致，不从合成未来预测创建新图。

修正记忆和原始正值图后，不能声称旧错误训练流程与新实现数值完全等价。对照测试明确只修正旧记忆读写，并在相同地理图和非负输入上检查有效网络的预测及梯度。地理图仍是平方距离权重，没有擅自改为距离越近权重越大的 Gaussian 核。

## 接口与目标

- 数据：`data.sampling.name=od_window`，`data.graph.name=weighted`（默认）或 `binary`。原始流量必须恰好一列，输出也是一列；日历/天气单独提供，不混入流量。GEML 保留原网络结构，不消费这些全局协变量。
- 模型：`model.params.embed_dim` 默认 32，`p_interval` 默认 1，均为正整数。周期大于历史长度时只分配实际会用到的状态。
- 输入：`X [B,Tin,N,N,1]`、同形布尔 `history_mask`。无需标签、scaler 或设备字段。
- 输出：`forward` 返回 `[B,H,N,N,1]`。训练期方法额外返回两个 `[B,H,N,1,1]` 边际预测，分别对应各起点的流出总量和各终点的流入总量。
- 训练：默认 `masked_mse`、`mask_value=null`，包含真实零流量。`training.loss` 中 `od_weight/origin_weight/destination_weight` 默认 0.5/0.25/0.25，可设非负权重且至少一个为正。默认 Adam、学习率 0.001、权重衰减 1e-6，100 轮、50 轮早停，不默认加调度器。

完整目标下，训练损失为上述三项原尺度 MSE 的加权和。出现 NaN 时，只有完整已知的一整行/列参与相应边际误差；有限零值仍进入真实边际求和。各项误差按有效 OD 单元广播并使用同一有效 OD 数作分母，保证不同微批和真实尾批的加权累积等价于一次完整有效批次。显式屏蔽零目标时也遵守这一规则，不把不完整的边际当作已知真值。

**验证与独立评估使用主 OD 的原尺度 MSE及公共预测指标，不计算辅助头。** `train_loss` 为复合训练目标，`val_loss` 为主预测误差，最佳权重按后者选取。这样保持无辅助分支的推理/评估接口及统一预测评测口径；这是相对旧执行器用复合损失验证的明确调整。辅助头在纯预测时不计算。

## 可运行示例

在仓库外生成数据和配置，所有路径以配置文件为基准：

```powershell
python <源码目录>/tools/make_runtime_fixture.py --directory <新目录> --model GEML
python <源码目录>/run_model.py train --config <新目录>/geml.json
python <源码目录>/run_model.py evaluate --config <新目录>/geml.json --checkpoint <新目录>/runs/geml-demo/checkpoints/best.pt --set runtime.exp_id=evaluated
python <源码目录>/run_model.py train --config <新目录>/geml.json --resume <新目录>/runs/geml-demo/checkpoints/last.pt --set runtime.exp_id=resumed --set training.epochs=3
```

发布配置为小规模 CPU 功能示例，不代表论文精度设置。[实际 CLI 记录](geml-cli-validation.json) 中四条命令全部成功，预测形状 `[9,2,3,3,1]`，训练后与独立评估预测/指标完全相等；第三轮累计 9 次更新，缓存 miss/hit/hit。独立评估仅加载安全纯权重，不创建训练记录、优化器检查点或 TensorBoard 目录。

## 验证

`tests/test_runtime_geml.py` 新增 36 项：四组修正旧记忆后的完整前向、全部有效梯度及辅助头；独立地理/语义图公式、孤立/超大距离；批次独立、单节点/单帧/float64、无标签和 RNG、跳过辅助头；三种 scaler、两种掩码、完整/缺失目标的逐单元独立损失及全部头梯度；实际带缺失值有效批次的 SGD 更新对照、全空目标；两种 scaler 的真实 OD 训练/独立评估及 Adam/metric 调度/RNG/最佳模型完整恢复；真实尾批 CPU AMP、Windows spawn 和无效配置。

结合 OD 与配置检查共 68 项通过，完整回归 **1045 passed，0 failed，0 errors，0 skipped**，记录在 [汇总报告](remaining-validation.json)。没有将这批 CPU 功能记录解释为全部城市、架构参数组合或 CUDA/MPS 精度验收。
