# STDN 原生迁移与时间对齐修复

STDN 已接入统一 train/evaluate/resume。保留局部流量门控 CNN、近期 LSTM、周期偏移 LSTM、内容注意力和高层周期 LSTM，使用标准 nn.Module、公共训练/原尺度评估、安全纯权重与完整续训。

## 原问题

旧 `_sample_stdn` 从较长的预热期之后构造局部特征，但 `_generate_input_data_stdn` 又用未加预热偏移的原始 volume 索引生成标签，辅助特征和真值时间不一致，可能把标签之后的观测作为输入。多步情况下按每个目标时间构造特征，并把 horizon 折进 batch，也不能代表在同一预测起点可用的输入。

旧原子流量读取假定 CSV 行已按起点/终点/时间排序；频率使用 `3600 // time_intervals` 再乘 24，不能准确表达所有日周期。划分按展开样本数，volume/flow scaler 基于展开窗口而非原始训练区间；局部窗口和向量一次性生成并重复保存在缓存中。网络还持有 scaler/logger/device，输出和损失依赖旧批次布局。

## 修复后的数据语义

所有输入以目标窗口起点 s 为同一观测截止：

- 近期序列为 `s-input_window .. s-1`，顺序从旧到新。
- 周期序列为前 A 天同刻附近的 L 个位置，从较早周期到较晚周期，L 要求为奇数；每个周期中心左右各 L//2 个时间片，且最近位置必须早于 s。
- 每个位置 t 的向量依次包含前 D 天同刻观测、t 之前最后 K 帧、t 的较小局部邻域，保留原通道和展平顺序。
- CNN 局部流量四通道依次为当前同时间段出流、入流，以及前一时间片跨时段出流、入流。所有位置均来自 s 之前，不从未来风险、流量或标签构造输入。

输入 X 的轴为 `[B,采样历史位置,N,局部行,局部列,volume特征]`，history_features 同步存放展平流量局部块和 volume 向量。采样顺序、空间身份、四流量通道和向量宽度均写入只读布局。通用输入提取器无需增加模型专用分支；y/y_raw 始终是独立的 `[B,H,N,output_dim]`。

`.geo` 必须给出完整矩形网格及唯一实际 geo_id；原始节点顺序可以乱序。`.grid` 或坐标式 `.dyna` 为 volume；`.gridod` 可用坐标或真实 origin_id/destination_id，按时间和端点显式对齐，要求完整时间×起点×终点记录，零流量明确写 0。flow_col 选择两个不同流量列，未给出时从表头推断；flow_files 可单独指定基名，否则沿用 data_files。拒绝重复/缺失关系、未知节点、负流量和不对齐的时钟/文件边界。

采用精确的 `86400 / time_intervals` 日周期，不能整除时在准备数据前明确报错。根据全部实际依赖（包括向量和上一时间片流量）推导最早有效目标；完整历史和目标跨度不跨文件或时间缺口。目标时间窗口严格互斥，历史可以使用切分之前已知的观测。

volume scaler 只从原始训练区间的目标列拟合，供 volume 输入和目标共同使用；flow_scaler 从原始训练区间的两类 OD 流量拟合。默认均为 normal；可显式选择其他公共 scaler。未观测的历史值仅在输入处补原始 0，边界也先补原始 0 再归一化，原始真值保留 NaN/0，损失和评估按 y_raw 判断掩码。

原始 volume 复用公共网格缓存，flow 单独缓存未展开数组；改变预测窗、周期/邻域、切分或任一 scaler 可复用原始缓存。局部窗口按样本切片，不持久保存展开后的块。来源只检查路径、大小和修改时间，不哈希文件内容；.gridod 来源变化会使完整续训拒绝恢复。仍保留原模型原始 OD 数组的 T×N² 内存成本，本批没有引入稀疏 OD 格式或覆盖第 8 节的全库数据深度重构。

## 保留的网络与有意变化

每个近期/周期时间位置各有独立参数，保留三层 64 通道 volume CNN 及门控。近期三层门分别从原始四流量通道计算，周期分支流量卷积逐层传递，均使用 ReLU 后 sigmoid；这个差别与[作者网络](https://github.com/tangxianfeng/STDN/blob/master/models.py)一致，未改成共享参数或另一类门。每个位置 flatten+dense 后，与对应向量拼接进入 LSTM。周期 LSTM 的所有输出由近期状态作 query 计算内容注意力，再按日顺序进入高层 LSTM，与近期状态拼接预测。

默认 input_window=7、H=1、三个周期/每周期三帧、D=7/K=48、CNN 半径 3/向量邻域半径 2、cnn_flat_size=lstm_out_size=128；两个 volume 特征时向量宽度为 160。设置属于命名 stdn_grid 协议，显式记录相对通用交通协议的变化。只读取原本使用的 volume/flow 上下文；外部及已知未来开关必须关闭，避免接受后静默忽略。未使用的 max_x_num/max_y_num/flow_type 不作为新结构参数保留。

H=1 的核心预测和全部参数/输入梯度与旧有效 Torch 网络对照。H>1 改成同一个预测起点的并行 H×output_dim 输出头，全部预测步接受监督，不按目标时刻读取未来特征。tanh 输出及默认原尺度 masked_mse（真实 0 掩码）保持；训练默认 Adam 0.001、100 轮、batch_size=1、早停 5，默认无调度/裁剪。

LibCity 的单层 PyTorch LSTM 使用 dropout=0.1 实际不生效，新实现去掉该无效参数和警告，没有宣称实现作者 Keras 的输入/循环 dropout，也不声称与旧错位数据流程或论文精度等价。

## 示例与验证

```powershell
python tools/make_runtime_fixture.py --directory ../stdn-demo --model STDN
python run_model.py train --config ../stdn-demo/stdn.json --set runtime.exp_id=train
python run_model.py evaluate --config ../stdn-demo/stdn.json --checkpoint ../stdn-demo/runs/train/checkpoints/best.pt --set runtime.exp_id=evaluate
python run_model.py train --config ../stdn-demo/stdn.json --resume ../stdn-demo/runs/train/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=resumed
```

示例显式缩短预热需求，近期 3 帧、前两天各 3 帧、D=1/K=3、两个邻域半径 1，预测 H=2；160 个小时、6 个网格、两个 volume 和两个 OD 流量列，原始数据无需下载。

42 项专项覆盖四组旧有效网络完整前向/全部参数与输入梯度（含默认 7/3/3/7×7/160/128）；独立嵌套循环的局部块、四种流向、历史向量与归一化；未来 volume/flow 隔离、真实 ID/空间置换、目标互斥/最后窗口、原始零/NaN；缓存无完整原子重读、三个损坏重建和七类无效来源；文件/缺口边界；两种 scaler 的实际训练/独立评估/原尺度 MSE 复算、所有预测步梯度、参数/优化器/metric 调度/最佳模型/RNG/加载器精确续训；两类真实尾批 CPU AMP、Windows spawn 和流量来源变更拒绝；14 类非法配置及命名协议。

完整回归 1225 项通过、无失败/错误/跳过，见 [汇总](remaining-validation.json)。公开示例的仓库外生成、训练、独立评估、续训 4 条实际命令成功，预测 [31,2,6,2]，独立预测/指标完全一致；第三轮累计 9 次更新、volume 与 flow 缓存均为 miss/hit/hit，原尺度 masked MSE 独立复算一致；纯权重包含输入说明，独立评估无训练产物，见 [CLI 记录](stdn-cli-validation.json)。本模型尚无真实城市精度复现及 CUDA/MPS 专项认证；现有 CPU 验收不替代这些结果。
