# 主线剩余问题执行记录

本轮依据根目录 runtime-architecture-plan_zh.md 和代码核查继续实施，基线为 `2a5ad72`，分支为 `refactor/libcity-modernization`。目标包括所有未延期模型的实际迁移、CRANN 完整数据语义与生命周期、基础缓存。方案第 8 节列出的延期事项保持延期；不把迁移清单或代表组合验收当成全库迁移完成。

已发布提交不改写；补充工作按可独立审查的功能提交。提交不添加 co-author，不做源文件内容哈希校验。

| 编号 | Commit | 提交标题 |
| --- | --- | --- |
| commit1 | 670b325 | fix: persist and reuse split-independent atomic traffic caches |
| commit2 | 956c1d5 | feat: migrate CRANN with explicit inputs and resumable shared training |
| commit3 | 23bf934 | feat: migrate six standard forecasts with native training contracts |
| commit4 | 1a8eb2e | fix: support CPU AMP when oneDNN convolution backward is unavailable |

## commit1：修复基础交通缓存（670b325）

问题：`data.cache=true` 原先仅创建目录，每次重复读取原子 CSV；基础预处理缓存未生效。

修改：缓存未归一化时序、已知未来协变量、时间戳、文件边界和邻接图。特征/图配置隔离缓存；根据源文件路径、大小和修改时间失效，不哈希源文件内容。切分、窗口、归一化和训练批次由各次实验重新计算。安全 NumPy 加载禁止 pickle；损坏文件重建；临时文件与原子替换防止读到半文件。运行摘要记录实际缓存状态与位置。

验证：`tests/test_runtime_traffic_cache.py`、`tests/test_runtime_timeline.py`、`tests/test_runtime_tuning.py` 共 **21 passed**。缓存测试验证命中不调用原子读取器、trial 参数变化复用原始缓存并重建窗口、数据/元信息/特征/外部协变量/图变化后的失效和隔离、损坏重建及禁用行为。

## commit2：完成 CRANN 数据语义与新运行时迁移

问题：旧路径缺少原始目标通道，仅验证目标互斥，未接入新训练入口；空间注意力固定为 24 步；batch=1 会被无条件 squeeze 破坏维度；网格列维被误当特征维，仅归一化第一列。

修改：新增标准 nn.Module，仅从 X 与显式 future_features 预测，外部未来协变量需通过 known_future_columns 声明；不读取 y，不持有 scaler/logger/device。data.sampling.name=crann 明确指定 temporal_history，空间历史仍由 input_window 决定；共享目标时间切分、训练区间归一化、y_raw 与原始缓存。temporal_channel 显式选择历史通道，dim_x*dim_y 必须覆盖节点且目标为单通道。注意力按实际 output_window 创建，保留 batch 维，隐状态随输入设备与类型初始化；n_layers_tem 实际控制编码与解码层数。旧参考数据路径同时保留 y_raw 并修复整网格归一化。

验证：CRANN 的 4 项测试及迁移表测试 **5 passed**；原版 24 步前向/梯度对照、无标签/batch=1、精确原尺度零掩码、目标时间互斥、真实原子文件短训练、独立评估和两轮连续/中断恢复参数逐项相等。提供 crann.json 与 --model CRANN 夹具生成命令。迁移范围更新为 16/100 条记录，84 条仍 pending，未声明全库完成。

## commit3：迁移六个标准交通模型

新增 AutoEncoder、AGCRN、TGCN、TrafficRNN、TrafficLSTM、Seq2Seq，均为标准 nn.Module，推理只接收输入字典，公共训练/评估/检查点组件承接训练职责。TGCN 稀疏邻接矩阵注册为 buffer，移除硬编码 device/float32；独立 step 保留原始训练正则目标。AGCRN 初始隐状态由输入创建。交通 RNN/LSTM 显式命名以避开轨迹 RNN 入口。

Seq2Seq 旧实现推理仍读取 y 且随机生成解码起点，现使用固定零起点，纯 forward 不读取标签、不消耗推理随机数；teacher forcing 由训练 step 显式传递归一化目标，交通 RNN/LSTM 同样通过显式训练方法实现。

验证：新增 21 项，逐模型对照旧前向及梯度（Seq2Seq 明确使用相同零起点）；TGCN 同时核对原正则目标；六个模型分别完整训练、独立评估及中断续训参数逐项一致；RNN/LSTM/Seq2Seq 覆盖两层 dropout 与随机 teacher forcing 的恢复；所有模型覆盖 batch=1、无标签确定性及 float64。与既有训练引擎合计 **40 passed**。清单更新为 22/100 已迁移，78 条仍 pending。

前三批完成后的完整回归为 **180 passed，0 failed，0 errors，0 skipped**。CRANN 发布示例实际 CLI 训练两轮、独立评估及续训至第三轮均成功；独立评估指标与训练后评估一致，评估/续训命中原始缓存。机器记录见 [remaining-validation.json](remaining-validation.json)。测试临时目录显式设置在工作区；系统沙箱临时路径下旧相对缓存测试曾遇到路径访问失败，同样测试在工作区独立临时目录 26/26 通过后，完整 180 项通过。

## commit4：修复 CPU AMP 卷积反向后端

新问题：当前 CPU/oneDNN 支持 BF16 卷积前向，却在反向报 `DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2`。STID、STNorm、ATDM 的实际 AMP 短训暴露了此公共问题；仅检查 AMP 前向无法发现。

修复：对 CPU AMP 卷积反向做一次实际小探测，隔离并恢复随机状态；仅后端不支持时，在反向阶段使用 PyTorch 原生卷积内核，保留 BF16 前向与 AMP。公共 backward 组件同时供默认循环、MTGNN、STSSL 使用；上下文退出恢复后端设置，不永久关闭 oneDNN，不修改 CUDA 路径。

验证：STGCN 实际 CPU BF16 训练、两轮连续/中断恢复逐参数相同及后端设置恢复；与既有定制训练、STSSL 测试共 12 项通过。下一批五模型的 AMP 测试也全部通过，随模型迁移独立记录。

## commit5：迁移日历、频域与归一化交通模型

新增 STID、STAEformer、FreTS、STNorm、ATDM，全部使用标准 nn.Module 与公共训练闭环，推理不读取标签。新增只读 calendar 通道元信息，按位置读取时刻与星期，避免新增观测列后索引错位。

明确修正的旧实现问题：STID 单独启用时刻/星期分支时嵌入维度交换；STAEformer 把星期 one-hot 的第一列当作星期整数，以及默认日采样数脱离元信息；FreTS 多变量 token 嵌入广播错误与多目标回归的维度重排；STNorm 单节点无偏方差导致 NaN、长输入残留多个时间位置及 n_pred 与预测窗/目标维度脱节；ATDM 回归头未包含全部目标维度。STNorm 的损失统一使用原尺度目标与公共掩码；FreTS 保留原始频域对角收缩，不更换数学算法。

验证：新增 **26 项通过**，包含五模型前向与梯度参考对照（STAEformer 使用正确的星期整数布局）、各自训练/独立评估/精确续训、多变量 batch=1/float64、CPU AMP；另外覆盖 STAEformer 两种投影、STID 独立日历开关和不同嵌入宽度、STNorm 单节点统计量。迁移表为 27/100 已迁移，73 条 pending，仍未完成全库迁移。

日历布局元信息扩充后，续训仍严格校验输入布局；旧布局不一致的 last.pt 会明确拒绝，纯 best.pt 权重评估仍须匹配模型结构。

前五批之后完整回归 **207 passed，0 failed，0 errors，0 skipped**，机器记录 [remaining-validation.json](remaining-validation.json) 保留此前 180 项结果的范围，并明确区分早前 CRANN CLI 记录和当前完整测试。

## 持续完成的清单

- [x] 基础交通缓存保存、读取、失效和独立目录。
- [x] CRANN 的统一输入、原始目标、完整训练/评估/续训与无标签推理。
- [ ] 全部未延期模型迁移、实际验证与迁移 CSV 更新（基线 15/100 已迁移；模板及第 8 节 Common 事项另行标注）。
- [ ] 扩大后的完整回归、发布示例与最终逐项验收；在此之前不声明主线全部完成。
