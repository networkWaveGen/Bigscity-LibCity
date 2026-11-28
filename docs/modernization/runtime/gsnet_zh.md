# GSNet 原生迁移与有意修订

GSNet 已接入统一 train/evaluate/resume，使用标准 nn.Module、风险数据适配、公共训练循环及区域评估。组件层由 commit52 单独交付，接口和损失公式见 [风险组件说明](risk-components_zh.md)。本批迁移原生网络、命名协议、配置校验与公开示例，不以 RiskProbe 代替实际模型。

## 保留的网络结构

保留 LibCity 的两个时空模块：

- 地理模块：每帧两层 3×3 网格卷积，逐网格 GRU，再用隐藏内容与预测时间特征计算历史时间注意力。
- 语义模块：road/risk 与可选 poi 各自的 GCN 堆叠，相加后逐图节点 GRU 和时间注意力，再映射回完整网格。
- 融合：两支各用 1×1 卷积投影到 16 通道，相加后用全连接层同时输出全部预测步。

图传播保留 `A @ X` 及原有 Linear/ReLU，不额外添加归一化或自环；数据配置可显式声明自环/对称化。图卷积广播静态邻接矩阵，时间特征按样本广播，不复制 batch/time 份图。没有 POI 图时不创建无效的 POI 参数。

默认参数为 5 层 GRU、隐藏宽度 256、GCN 宽度列表 [64,64]；修复旧默认把列表设成整数 64 的错误。输入宽度来自实际观测、历史全局特征和选中图观测；模型不持有 scaler/logger/device。图、网格映射和图特征索引均为独立 buffer，跟随标准 device/dtype。

## 修复的预测与损失语义

旧包装交换矩形网格的行列，并把预测步放在最后一维。损失再按 output_dim=1 对最后一维切片，结果是多步预测只监督第一步。新前向固定返回 `[B,H,N,1]`，N 是完整行优先网格；风险损失独立监督所有 H 步，原尺度真值及区域掩码由数据层提供。

旧网络用历史第一个时间片作为 target_time_feature。新网络只从 `future_features[:,0]` 取得第一个真实预测时刻的已知信息，训练和推理均不读取 y/y_raw。原融合头只接收一个预测时间向量并同时输出 H 步，这个结构保持不变；后续预测时刻的已知向量不会另外进入网络，不把它描述为逐时刻自回归模型。

注意力只 squeeze 最后一维，保留 batch、节点和时间轴；单样本、单格、单历史帧均可运行。未启用任何已知未来特征时，省去时间投影层，使用隐藏内容注意力；没有外部特征时只需 X。这两种可选模式不创建无用参数。

原始风险分档始终按原尺度解释，默认从数据元信息读取，可通过 `training.loss.thresholds/weights` 覆写。真实零风险参与训练和评估，mask_value 必须为 null；区域 mask>0 决定评估资格，幅度的平方作用仅用于风险训练目标。原作者实现曾使用归一化标签的分档，本实现不声称该目标与作者归一化阈值或论文精度等价。

## 配置与数据协议

命名协议 risk_cpt 默认最近 3 帧、前 1～4 周同刻共 4 帧，trend=0，默认预测 1 步。有效历史长度按激活偏移计算，而不是直接使用公共任务默认 input_window；偏离公共评测协议及有效长度均写入日志/运行摘要。默认长周期需要足够长的训练前缀，小型示例显式缩短为最近 2 帧和前一天同刻 1 帧。

`.geo` 是部分图节点，要求唯一实际 ID、row_id/column_id 和 risk_mask；config.json 的 grid_len_row/grid_len_column 定义完整网格。`.grid` 或坐标式 `.dyna` 提供完整网格，`.rel` 的路图/风险图/可选 POI 列按真实图 ID 对齐。data.graph 的 road_column/risk_column/poi_column 可覆写元信息 weight_col，use_poi=false 可显式关闭 POI。这些图作为给定静态先验使用，不从全时间轴重新估计。

data.features.graph_columns 显式选择历史图观测，未给出时采用元信息 graph_input_col 或首个目标；不能把未观测的未来风险当图特征。历史外部数值列逐列用原始训练区间拟合 external_scaler，默认 minmax01；24 小时/7 星期 one-hot 保持原义。未来仅含日历和 known_future_columns 明确声明的列。缓存、时间切分、身份与输入说明、完整续训来源检查复用公共能力。

训练默认 Adam 1e-5、最多 200 轮、batch_size=32、早停 patience=10/min_delta=1e-6；默认风险 MSE，监控 val_loss 最小化。默认区域指标为 RMSE、Recall、MAP，topk=10；也支持公共误差、Precision、F1-Score、PCC 及 single/average 模式。排名精确取 K、按网格索引解并列，零正例和退化相关返回 0，具体定义见组件说明。

## 可运行示例与验收

在源码目录使用：

```powershell
python tools/make_runtime_fixture.py --directory ../gsnet-demo --model GSNet
python run_model.py train --config ../gsnet-demo/gsnet.json --set runtime.exp_id=train
python run_model.py evaluate --config ../gsnet-demo/gsnet.json --checkpoint ../gsnet-demo/runs/train/checkpoints/best.pt --set runtime.exp_id=evaluate
python run_model.py train --config ../gsnet-demo/gsnet.json --resume ../gsnet-demo/runs/train/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=resumed
```

新增 33 项专项覆盖四组原 LibCity 核心模块的完整前向和全部参数/输入梯度（包括默认 5/256/[64,64]）；矩形网格、双图/三图、独立图节点置换、buffer 所有权、单样本/单格/单帧/零外部/零未来特征、多步头梯度、无标签及跨样本隔离、真实未来时间上下文；两种 scaler 的实际训练/独立评估、原尺度风险目标和区域误差独立复算、优化器/metric 调度/早停/最佳模型/RNG/加载器精确恢复；双/三图真实尾批 CPU AMP、Windows spawn，以及 19 类非法配置启动拒绝。相关组件和配置检查合计 73 项通过。

实际发布例子的生成、训练、独立评估、续训 4 条 CLI 命令成功，预测形状 [19,2,6,1]，两次预测和指标完全一致，第三轮累计 12 次更新，缓存 miss/hit/hit；原尺度区域平方分档 MSE 独立复算一致。独立评估无训练产物，best.pt 保持安全 Tensor state_dict 并携带输入说明，见 [CLI 记录](gsnet-cli-validation.json)。完整回归 1183 项通过、无失败/错误/跳过，见 [汇总](remaining-validation.json)。

本批为 CPU 功能、数值参考和生命周期验收，未执行真实事故数据精度复现或本模型 CUDA/MPS 认证。保留原模型稠密 G×G 图和 16N→HN 全连接头的成本；未把这些结构变化或论文精度优化隐含加入迁移。
