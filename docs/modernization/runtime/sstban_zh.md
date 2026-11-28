# SSTBAN 原生迁移与显式自监督训练

SSTBAN 已接入统一 train/evaluate/resume。保留可学习节点嵌入、时空瓶颈注意力、历史到未来的变换注意力、编码/解码残差与自监督表示一致性任务。普通 forward 接收标准输入字典并只返回 [B,H,N,O]；forward_with_aux 由训练步显式传入保留掩码，返回预测、完整编码和受损编码。

## 原问题与修复

旧模型在内部生成 NumPy 随机掩码，通过 mode 字符串切换路径，并将 X_miss/complete_X_enc 留在对象上供之后的损失读取。注意力按 K 而非 d 个通道拆分，默认 K=16、d=8 实际得到 8 个宽 16 的头，却仍除以 sqrt(8)。掩码按 query/key 形状猜测作用对象，当瓶颈参考数等于节点数或时间长度时有歧义；全遮蔽行填相同 -1e9 后变成均匀注意力，低精度也可能溢出。掩码块使用 floor(T/block_length)，短历史或非整除会产生长度错误。

新实现显式保留 B/K 两条轴，使用 K 个宽 d 的头及正确缩放；MAB 的第一个瓶颈步骤明确约束 key，第二个明确约束 query。标准 scaled_dot_product_attention 接收布尔掩码，全遮蔽行的注意力加权结果为零，后续 FC/BN 仍按模型定义执行。避免沿 batch 拼接/拆回头和实体形状推断；诱导点使用 expand，不复制 batch 份参数。三条时空残差路径及逐点 Conv2d/BN/ReLU 顺序保持。

掩码生成移入训练步，用设备上的 Torch RNG，按 ceil(T/block_length) 采样后重复并裁到 T，支持短历史和最后不足一块。保持[作者代码](https://github.com/guoshnBJTU/SSTBAN/blob/main/model/sstban_model.py)在潜在通道上采样、共享完整/受损编码器、在自监督解码块前替换 mask_token 的次序。一个头内至少保留一个通道，该节点/时刻就对该头有效。原名 node_miss_rate 实际控制 mask=1 的保留概率，因此配置明确改为 training.strategy.keep_probability；True 表示保留，False 表示遮蔽。原 T_miss_len 改为 block_length，不让训练随机策略进入模型构造。

辅助损失仍比较完整/受损表示，受损表示精确非零的分量有效，两边都保留梯度；不引入未声明的 stop-gradient。每个样本先算有效潜在分量的 MSE，再按其有效预测目标数量加权，与主任务共用有效数量参与梯度累积。主损失为原尺度 masked_mae(0)，训练混合 `(1-self_weight)*主损失+self_weight*辅助损失`。这是明确修订的批次聚合口径，避免原整批隐表示平均与目标计数冲突；257/259 等计数不经过 BF16。全无有效目标时不贡献辅助项，零隐表示返回可微零值；非有限隐表示明确拒绝。

验证、最佳权重选择和独立评估只使用主预测损失，不生成随机掩码、不执行自监督块。self_weight=0 时训练也跳过采样及辅助前向。模型没有 scaler/logger/device、mode 或跨调用暂存表示；单值 BN 复用公共运行统计回退，保留真实尾批。

## 数据与配置

公共 CalendarWindows 增加可选未来日历，原 DMSTGCN 历史日历默认路径保持。SSTBAN 使用 prepare_calendar_forecast：复用实际 geo_id/UTC 对齐、原始缓存、目标窗口互斥切分和训练区间 scaler，历史 NaN 原尺度补零，NaN/0 目标保留。按原子 clock 按需切出 int64 history_features/future_features，均为 [B,T或H,2] 的星期（周一=0）/日时间槽。日期与未来时刻是已知信息，未来观测不作为输入。时间间隔必须为一天的正整数约数，时间槽自动跟随实际频率。

不再只读取第一观测/第一目标，也不再只保留最后一个 data_file；公共数据层支持多文件按时间顺序与多变量数据，历史可读取外部列。输入和输出宽度从元数据确定，不再接受 in_channels/out_channels。模型不使用原子路图，data.graph.name=none 且无需 .rel。专用整数日历已经提供时间信息，因此 data.features.time_in_day/day_in_week 及 known_future_columns 必须关闭，避免把浮点日历或其他未来协变量混入同一接口；load_external 可独立开启历史外部观测。

默认 L=3、K=16、d=8、reference=3、bn_decay=0.1。保持共享任务的切分/窗口/评估规则，不沿用旧特殊 Dataset 的 36→36 窗口覆盖；长预测可由用户显式设置。默认 100 轮、batch_size=4、Adam 0.001/eps=1e-8/weight_decay=0.0001、早停 5、self_weight=0.8、block_length=12、keep_probability=0.5。默认无调度，保持旧 Executor 的 lr_decay=false 实际行为；如需旧配置中列出但未启用的 StepLR，可显式选择 step/epoch、step_size=10、gamma=0.9。

```powershell
python tools/make_runtime_fixture.py --directory ../sstban-demo --model SSTBAN
python run_model.py train --config ../sstban-demo/sstban.json --set runtime.exp_id=train
python run_model.py evaluate --config ../sstban-demo/sstban.json --checkpoint ../sstban-demo/runs/train/checkpoints/best.pt --set runtime.exp_id=evaluate
python run_model.py train --config ../sstban-demo/sstban.json --resume ../sstban-demo/runs/train/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=resumed
```

示例有 60 小时、4 个非连续真实 ID 节点、3 个历史通道、2 个目标和实际 24 槽日历，无 .rel；历史 5 步预测 3 步，block_length=3 覆盖不足一整块的尾部。小模型 L=1、K=2、d=3、reference=2，self_weight=0.4、keep_probability=0.6。best.pt 仅含 Tensor 权重及输入说明；last.pt 保存优化器/调度器/BN/最佳模型/Torch RNG/加载器等完整恢复状态。

## 验证与范围

新增 35 项专项，四组修正旧分头/掩码/时间 dtype 后的完整预测/辅助表示/全部参数及输入梯度/BN（含默认 L3/K16/d8、K=d、歧义参考数、多变量与单节点），独立逐头注意力及全部梯度、明确 query/key 掩码与空邻域零结果，保留概率 0/0.4/1、非整除/短历史块，两种 scaler 的随机掩码/BN/Adam/metric 调度/早停/最佳模型/RNG/加载器精确恢复，独立辅助 MSE 与 257/259 目标微批梯度，原尺度 NaN/0、真实日历/缓存/未来隔离/无标签无 RNG、普通推理不执行辅助块、零权重跳过辅助工作、全遮蔽 CPU AMP/真实单样本尾批、Windows spawn 与 15 类配置错误。DMSTGCN/配置相关合计 80 项通过，完整回归 **1427 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。

仓库外实际生成、训练、独立评估和续训 4 条命令通过，预测 [10,3,4,2]，独立预测/指标完全一致，第三轮累计 9 次更新；缓存 miss/hit/hit、24 槽日历和无 .rel 场景均验收。原尺度 masked MAE 复算约 7.410293 与评估一致，纯权重安全加载，独立评估没有训练产物，见 [机器记录](sstban-cli-validation.json)。对分头、掩码角色、全遮蔽注意力和辅助聚合均明确修订，不声称与旧错误实现逐值一致或复现论文精度。本批验证 CPU 功能与恢复；真实城市精度、CUDA/MPS 专项和长窗口显存基准另行记录。
