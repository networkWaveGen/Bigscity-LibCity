# D2STGNN 原生迁移与显式课程学习

D2STGNN 已接入统一 train/evaluate/resume。保留估计门、扩散/固有信号解耦、局部时空卷积、GRUCell/多头时间注意力、两类 backcast、动态图与学习静态图，以及 gap 分组预测。模型为标准 nn.Module，forward 只接收历史输入并返回 [B,H,N,O]；课程监督由训练函数组合公共 train_loop 完成。

## 原问题与修复

旧模型内部持有 scaler、device、num_batches 与可变 task_level；续训无法只依靠模型权重正确恢复课程阶段。node_dim 与内部 node_hidden 使用不同配置键，非默认维度不匹配；预设 time_emd_dim 拼写也未实际控制 time_emb_dim。时间表固定为 288，列位置根据尾部 8 列猜测，日历后还有外部变量时会错取输入。图张量未注册 buffer，在每层重复保存；局部图为每个时间位置重复为 [N,k_t×N]，在预测递推中再次准备相同支持。

新模型统一使用 node_dim/time_emb_dim，显式声明 use_time_of_day/use_day_of_week 与 attention_heads，按只读元数据定位日历列。数值历史取全部非日历列，包括日历后面的历史外部变量；dynamic_channel 默认取其中第 0 列驱动动态图。时间槽默认按 86400/time_intervals 推导，也可显式 steps_per_day；不整除的自动槽数在构建时拒绝。索引保留浮点边界容差，修复精确小时等位置落到前一槽；星期仍用 7 维 one-hot。关闭的时间表不创建参数。

预定义路图保留 transpose(D^-1 A) 与 transpose(D^-1 A.T) 的原方向，动态图分别以 d/u 节点嵌入、最后历史日历和第一指定数值序列计算 Q/K 相似度，再乘对应路图+1e-7、行归一化。静态图保留 softmax(relu(E_u@E_d.T))。各阶均先计算完整矩阵幂再去对角，不能在计算下一阶前去对角。原 self 分支为时间窗口投影后的均值，其他图分支保持原求和尺度。

路图、预定义幂和对角掩码注册独立 buffer；静态/动态图幂每次 forward 只准备一次，供所有解耦层及递推复用。局部卷积利用 `[A,...,A]@[X_1;...;X_k]=A@sum_t(X_t)`，省去时间重复图及 k_t×N 空间展开，保留先沿时间拼接线性投影/ReLU、self 均值、按支持顺序拼接和输出线性/Dropout。低精度图分数、归一化和传播使用 float32。动态图特征 BN 复用公共单值回退。

预测分组改为 ceil(H/gap)，最后裁到 H；H 小于 gap 时实际组宽取 H，避免多余输出行。保证 H=1、非整除、多目标都获得明确形状和监督。全部时间卷积层必须满足 input_window>=1+num_layers×(k_t-1)，隐藏维度须能被注意力头数整除；奇数隐藏宽度的一维头位置编码也可运行。

固有模块保留[作者 forecast](https://github.com/GestaltCogTeam/D2STGNN/blob/github/models/inherent_block/forecast.py)中的递推 GRU、再次位置编码与查询顺序。注意力不请求未使用的权重矩阵，位置 buffer 仅按实际历史/预测分组长度创建。删除未使用的 time_slot_embedding、局部卷积 BN，以及最终解耦层无后续消费者的固有 backcast/LayerNorm；扩散 backcast 仍是固有分支的有效输入。原始动态特征 BN、位置 Dropout、预测递推与所有有效层均保留。

## 课程学习与数据语义

training.strategy.use_curriculum_learning=true 时，第 e 个完整 epoch 开始监督 `min(H,e//cl_decay_epochs+1)` 个预测步，e 从 0 开始。训练步将阶段写入 TrainState.task_state.d2stgnn_curriculum_horizon，状态随 last.pt 保存；阶段由已完成 epoch 推导，不再依赖模型内部计数或手填 num_batches。对固定 epoch 的完整加载器，它与旧每 cl_decay_epochs×num_batches 增加一步的时间表一致。改变课程配置续训会被公共设置检查拒绝。

训练损失默认原尺度 masked_mae(0)，仅计算当前课程前缀，并按有效目标数量参与梯度累积。验证、最佳权重选择、训练后测试和独立评估始终使用全部 H 步，普通 forward 不持有/推进课程阶段，不依赖 y 或真实未来观测。原始 NaN/零值目标保留。

HIEST、STMGAT 和本模型的历史补零提取为公共 impute_history/prepare_finite_traffic，减少重复的数组与元数据处理；补零仅作用于历史副本，在原尺度补 0 后使用训练区间 scaler，原始目标与缓存不改写。原子身份/UTC 对齐、切分、缓存和来源校验继续复用点数据层。模型不额外使用 future_features，历史外部变量可通过 data.features.load_external 选择读取。

## 配置与使用

默认 hidden=32、node_dim=10、forecast_dim=256、output_hidden=512、k_s=2、k_t=3、5 层、time_emb_dim=10、gap=3、4 个注意力头、dropout=0.1；启用动态/学习静态图和双日历，关闭预定义图分支。动态图仍使用原子路图作掩码。可独立关闭三种图；graph.name=none 不读 .rel，开启动态图时仍按 epsilon 掩码构建学习关系。共享切分/窗口/评估协议不被预设覆盖。

默认 100 轮、batch_size=32、Adam 0.002/weight_decay=1e-5/eps=1e-8，epoch MultiStep([1,30,38,46,54,62,70,80],gamma=0.5)、裁剪 5、早停 30、每 3 个 epoch 扩大监督一步。

```powershell
python tools/make_runtime_fixture.py --directory ../d2stgnn-demo --model D2STGNN
python run_model.py train --config ../d2stgnn-demo/d2stgnn.json --set runtime.exp_id=train
python run_model.py evaluate --config ../d2stgnn-demo/d2stgnn.json --checkpoint ../d2stgnn-demo/runs/train/checkpoints/best.pt --set runtime.exp_id=evaluate
python run_model.py train --config ../d2stgnn-demo/d2stgnn.json --resume ../d2stgnn-demo/runs/train/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=resumed
```

示例为 60 小时、4 个真实 ID 节点（有向环与孤立节点）、3 个数值历史列、双日历和 2 个目标。小网络两层、hidden=4、2 头、node/time_dim=3、gap=2，6 步历史预测 3 步，课程每 epoch 扩大一步。纯权重安全保存输入说明、路图/位置 buffer 和实际 24 槽时间表，独立评估不创建训练产物。

## 验证与范围

新增 36 项专项：七组旧网络预测/全部有效参数和输入梯度/BN（含默认五层宽度、多目标、非整除、各类图/日历与纯 self 模式），三类局部时间核的原始展开图公式/所有梯度对照，独立方向/矩阵幂与对角规则、显式课程/有效数量/无监督步零梯度、两种 scaler 的课程切换/dropout/BN/Adam/metric 调度/早停/最佳模型/RNG/加载器精确续训，批次独立/静态所有权/float64/无标签无 RNG、小时槽与中间日历后外部列、原始 NaN/0/缓存/未来隔离、真实单样本尾批 CPU AMP/奇数隐藏宽度/单节点、Windows spawn、课程变更拒绝和 16 类配置错误。包含 HIEST、STMGAT 与配置的相关回归 123 项通过；完整回归 **1392 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。

仓库外实际生成、训练、独立评估和续训 4 条命令通过，预测 [10,3,4,2]，独立预测/指标完全一致；第三轮累计 9 次更新，课程监督达到第 3 步，缓存 miss/hit/hit。原尺度 masked MAE 独立复算约 8.709683，与评估一致；安全纯权重含 24 槽时间表，独立评估无训练产物，见 [机器记录](d2stgnn-cli-validation.json)。本批验证 CPU 功能和恢复，使用相同参数及明确补齐 gap 的旧参考，不宣称旧非整除输出可直接使用或复现论文精度；稠密动态图/矩阵幂和时间注意力的成本仍保留，真实城市精度与 CUDA/MPS 专项另行记录。
