# STMGAT 原生迁移与有向边注意力

STMGAT 已接入统一 train/evaluate/resume。保留双路输入投影、门控膨胀时间卷积、每个有效层两次多头图注意力、图/时间残差、BN 与 skip 输出。forward 为标准 dict[str,Tensor] → [B,H,N,O]，不读取 y、不持有 scaler/logger/device。

## 原问题与修复

旧实现依赖 DGLGraph、DGL 批图及 GATConv；空关系表会在 zip 解包时报错，自环可能重复添加。门控、skip 与残差使用 Conv1d 接收四维 Tensor，当前 PyTorch 无法执行。图注意力宽度由预设感受野推算，input_window 超过感受野时实际 C×T 不匹配；感受野初值使用 output_dim，又把最后剩余时间维直接当成目标维。图分支相加隐含 residual_channels=dilation_channels，却没有提前校验。最后一层在 skip 后提前结束，仍创建无效的 GAT/BN/残差参数。

新实现以原生 PyTorch 有向边索引替代 DGL。A[i,j]>0 表示 i 向 j 发送消息，权重幅值不乘入注意力；旧代码虽然保存 w，但调用 GATConv 时也未传 edge_weight。去除原有对角边后给每节点补一个自环，空图和孤立节点都有合法入邻域，显式修复重复自环造成的额外权重。边索引仅保存一份 buffer，批次保持独立轴，不再构造 B 份批图。

每头保留完整 C×T 投影宽度。按源/目标投影分数相加、LeakyReLU、目标节点入邻域 softmax、attention dropout、加权求和、逐头 bias/ELU、最后头平均的顺序执行；连续两层都按此顺序。实现对照 [DGL 官方 GATConv](https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/gatconv.py)。逐目标/逐头最大值防止无关邻域的大分数导致下溢，低精度投影后的注意力/聚合在 float32 中完成。保留 feature dropout 与 attention dropout；旧 dropout 字段从未参与计算，新的配置不再接受这个无效开关。

全部时间层改用 Conv2d，感受野为 1+blocks×(kernel_size-1)×(2^layers-1)，每个 block 重置膨胀率。左侧补零到至少感受野；GAT 宽度按 max(input_window,receptive_field) 及各层实际收缩推导，长历史不裁断、不改变原 flatten(C,T) 含义。skip 只累计最终预测时刻的位置，预测头显式生成 H×O 后恢复目标轴。多步、多目标均受监督，不再借用剩余历史时间位置表示不同目标。

启用图分支且总层数大于一时要求 residual_channels=dilation_channels，原加法结构保持；关闭图分支后使用原 1×1 残差投影，允许两种宽度不同，可选 data.graph.name=none 且无需 .rel。总层数为一时最后图原本就不参与预测，保留此时间预测结构。只创建实际使用的 GAT/残差/BN，公共单值 BN 回退保留真实单样本尾批，不伪造重复观测。

数据使用公共点窗口、真实节点 ID 与 UTC 对齐、目标时间互斥切分、训练区间归一化和原始缓存。仅历史 NaN 在原尺度补 0 后再归一化，NaN/0 原始目标保留；无限观测拒绝。未来标签或实际未来观测不进入模型，已知未来特征也不额外接入本网络。模型预设不改任务基准切分、窗口或评估掩码。

默认损失为原尺度 mse，仅排除非有限目标，保留真实零值。这与旧 masked_mse_torch 默认 null_val=NaN 的有效目标含义一致；不再原地把绝对值小于 1e-4 的真实目标改成零。用户可显式选择 masked_mse 及 mask_value。训练损失和公共 MAE/RMSE 评估的掩码分别以 training.loss/evaluation 声明，不把损失默认值悄悄改成统一评估规则。

## 配置与使用

默认 blocks=4、layers=2、kernel_size=2、run_gconv=true，residual/dilation=40、skip=320、end=640、heads=8、feat_drop=0.6、attn_drop=0.6、negative_slope=0.2；Adam 0.001、100 轮、裁剪 5、无调度、早停关闭。支持连续 window 的多变量点预测，自环与正值连接规则由上述网络定义。

```powershell
python tools/make_runtime_fixture.py --directory ../stmgat-demo --model STMGAT
python run_model.py train --config ../stmgat-demo/stmgat.json --set runtime.exp_id=train
python run_model.py evaluate --config ../stmgat-demo/stmgat.json --checkpoint ../stmgat-demo/runs/train/checkpoints/best.pt --set runtime.exp_id=evaluate
python run_model.py train --config ../stmgat-demo/stmgat.json --resume ../stmgat-demo/runs/train/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=resumed
```

发布示例有 60 个小时、4 个非连续 ID 节点（有向三节点环与一个孤立节点）、3 个历史通道和 2 个目标，.geo 行打乱。小模型一个 block/两层，4 个时间通道、3 个注意力头，历史 4 步、预测 2 步。安全 best.pt 包含 Tensor 权重、edge_index 和输入说明；last.pt 保存优化器、调度器、BN、最佳模型、RNG/加载器等完整状态。

## 验证与范围

新增 31 项专项，覆盖独立逐入邻域注意力公式/全部梯度，五组旧时间/融合代码结合独立 DGL 公式的完整输出/梯度/BN 参考（含默认 40/320/640/8、长历史、多目标、异宽无图与单层），源到目标方向、权重幅值、一次自环、空图/孤立节点、局部稳定 softmax、节点置换与跨样本隔离、float64、无标签无 RNG，原始 NaN/零值损失、缓存/未来隔离，两种配置的 dropout/BN/Adam/metric 调度/早停/最佳模型/RNG/加载器精确续训，禁止 DGL 导入后的实际 CPU AMP、真实单样本尾批、Windows spawn、无 .rel 时间分支及 15 类配置/图错误。相关配置与公共节点身份合计 61 项通过；完整回归 **1356 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。

实际仓库外生成、训练、独立评估和续训 4 条命令通过，环境未安装 DGL，预测 [11,2,4,2]；独立预测和指标完全一致，第三轮累计 12 次更新，缓存 miss/hit/hit。原尺度 MSE 独立复算约 57.420934，与评估一致，纯权重安全加载、7 条边及独立评估无训练产物均验收，见 [CLI](stmgat-cli-validation.json)。

对旧不可运行的 Conv1d 与长历史/多目标分支使用明确修正后的参考，未声称直接运行原 DGL 模型或复现论文精度。注意力参数规模仍按 heads×(C×T)^2 增长，没有用拆小每头宽度替换原算法；本批完成 CPU 功能，真实城市精度与 CUDA/MPS 专项认证分别记录。
