# DSAN 原生网格迁移

DSAN 通过连续网格数据、标准 nn.Module、默认 train_loop 和公共 RMSE 目标接入统一训练、独立评估与完整续训。生产代码不调用旧 AbstractTrafficStateModel，也不在模型内保存 scaler、logger 或 device。

## 原问题与修复

| 原问题 | 处理 |
| --- | --- |
| 局部区域折叠进 batch 后，普通 repeat 将其他样本的全局上下文混入当前节点 | 显式保留 B、N、时间、注意力空间轴；全局上下文只在同一样本的区域维广播，不复制其他样本 |
| 局部卷积在重排后的全局网格上运行，局部偏移、中心节点和通道次序混杂 | 用 pad/unfold 提取每个中心的真实局部方块，卷积在方块内部进行；每个历史位置仍保留原来独立的一组卷积参数 |
| Enc-D、Dec-S、Dec-T 每层都读取相同初始输入，前面层输出被丢弃 | 三类堆叠均逐层传递输出；所有有效参数均进入预测与反向传播 |
| 拆分多头后又除一次头数，错误缩放注意力；因果掩码连对角也遮蔽，首行为空时仍产生均匀注意力 | 按实际 head_dim 开方缩放，对角可见；全遮蔽行先安全处理，再将注意力聚合置零，避免 NaN 和读取被遮蔽键 |
| FFN 写死 64、种子映射写死两列、最后 reshape 交换矩形网格行列 | 由 d、实际目标数和只读网格位置推导维数，预测恢复真实 `.geo` 节点顺序 |
| 掩码基于归一化值求和，真实零值或正负抵消会改变语义；位置编码每批创建并固定设备 | 在归一化前生成有限且非零的原始观测布尔掩码；局部/全局位置、因果掩码及节点映射均注册为 buffer |
| 预测入口读取 y，RMSE 在不同微批/尾批上的平均不等于整个有效批次目标 | 前向仅接收输入；复用公共整组 RMSE 梯度归一化和跨批评估，并提取共用 loss 构造器供 DSAN/CCRNN 使用 |

缩放、可见对角和逐层传递核对了[作者实现](https://github.com/haoxingl/DSAN/blob/master/models.py)及[作者掩码](https://github.com/haoxingl/DSAN/blob/master/utils/tools.py)。本次交付保留 LibCity 的全局/局部编码、时间到预测窗的线性映射、空间/时间切换解码和 tanh 输出；修正上述实现错误，不声称旧错误运行数值或完整论文精度等价。

## 数据与前向接口

新增 `data.sampling.name=grid_window`，与 CPT 模式区分。支持 `.grid` 或带网格 ID 的 `.dyna`，按真实 `.geo` 行序提供 `[B,Tin,N,F]`，历史偏移为 `[-Tin,...,-1]`。目标 `[B,H,N,D]` 取前 D 列，历史可越过集合切分点使用已知前序观测，目标时间集合互斥；scaler 只拟合原始训练区间。

- 必需输入：`X` 和同形 bool `history_mask`。掩码在原始数值上判断有限且非零，不因归一化偏移、NaN 补零或通道求和抵消而改变。
- 开启全局外部特征时另有 `[B,Tin,E]` 的 `history_features`；其与历史非目标观测列共同进入时间编码。没有外部列时省略对应编码器。
- `forward` 始终返回 `[B,H,N,D]`，支持单格、单样本、多目标和不同预测长度，不读取 y/y_raw。
- 保留 LibCity 的历史解码种子：H 不大于 Tin 时使用最近 H 帧；超过时在已有历史后重复最后一帧，明确作为历史种子长度扩展。外部历史和原始掩码使用同样规则。
- 此版本采用 LibCity 的历史种子并行解码，不使用 `future_features`，也不复制作者训练时的真实未来 teacher forcing。它不是作者逐步自回归流程的精度复现。

数据层不要求静态关系图：`data.graph.name=none`，不读取 `.rel`。基础缓存仍仅存未展开数组，切分、窗口、scaler、批次变化可复用；命中时不重读完整观测/外部数据，只读必要元信息/表头。补充数值类型、有限性、时钟和文件区间检查，坏缓存重建。连续窗口允许不整除一天的采样频率；原 CPT 模式的顺序和周期规则保持。

## 配置与运行

默认模型参数为 `L=3,d=64,n_h=8,B=3,l_d=2,r_d=0.1`。L 控制注意力堆叠，B 控制卷积/种子映射深度，l_d 是局部半径，d 必须能被 n_h 整除。默认原尺度 masked_rmse 排除真实零目标；Adam 0.0001、eps=1e-9，按 epoch 在 5/20/40/70 轮降低学习率，100 轮、50 轮早停。输入窗/预测窗沿用任务基准默认值，可显式覆写并记录协议变化。

```powershell
python <源码目录>/tools/make_runtime_fixture.py --directory <新目录> --model DSAN
python <源码目录>/run_model.py train --config <新目录>/dsan.json
python <源码目录>/run_model.py evaluate --config <新目录>/dsan.json --checkpoint <新目录>/runs/dsan-demo/checkpoints/best.pt --set runtime.exp_id=evaluated
python <源码目录>/run_model.py train --config <新目录>/dsan.json --resume <新目录>/runs/dsan-demo/checkpoints/last.pt --set runtime.exp_id=resumed --set training.epochs=3
```

[实际发布示例](dsan-cli-validation.json)四条仓库外命令成功；预测 `[11,2,6,2]`，独立预测和指标完全一致，第三轮累计 12 次参数更新，缓存 miss/hit/hit。安全 best.pt 带公共输入说明，独立评估不创建训练产物；全局原尺度 masked RMSE 由导出预测独立核对。

## 验证

新增 37 项：四组使用旧有效 Torch 子模块、按样本/区域独立执行的修正参考，比较完整预测及全部参数/流量/外部梯度，包含默认宽度/堆叠及非 64/非双目标维度；注意力与 PyTorch SDPA 的值/梯度/全遮蔽对照；NumPy 奇数/偶数位置编码；局部方块的值、布尔掩码及梯度；矩形节点置换、跨样本零梯度、无标签/未来输入/RNG、长预测窗；原始零/NaN/训练区间统计与未来隔离；原始缓存无完整行重读及四种损坏重建；非整日频率；两种 scaler 的 dropout/Adam/metric 调度/全状态精确续训；真实尾批 CPU AMP、单格单帧和 Windows spawn。

与原网格、根损失及 CCRNN 合计 143 项通过。参考前向在关闭 dropout 时核对有效网络；随机训练只声明新实现自身的完整续训等价。完整回归 **1122 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)，不据此认证所有参数组合、真实城市精度或 CUDA/MPS。
