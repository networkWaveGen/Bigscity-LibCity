# 主线剩余问题执行记录

本轮依据根目录 runtime-architecture-plan_zh.md 和代码核查继续实施，基线为 `2a5ad72`，分支为 `refactor/libcity-modernization`。目标包括所有未延期模型的实际迁移、CRANN 完整数据语义与生命周期、基础缓存。方案第 8 节列出的延期事项保持延期；不把迁移清单或代表组合验收当成全库迁移完成。

已发布提交不改写；补充工作按可独立审查的功能提交。提交不添加 co-author，不做源文件内容哈希校验。

## commit1：修复基础交通缓存（670b325）

问题：`data.cache=true` 原先仅创建目录，每次重复读取原子 CSV；基础预处理缓存未生效。

修改：缓存未归一化时序、已知未来协变量、时间戳、文件边界和邻接图。特征/图配置隔离缓存；根据源文件路径、大小和修改时间失效，不哈希源文件内容。切分、窗口、归一化和训练批次由各次实验重新计算。安全 NumPy 加载禁止 pickle；损坏文件重建；临时文件与原子替换防止读到半文件。运行摘要记录实际缓存状态与位置。

验证：`tests/test_runtime_traffic_cache.py`、`tests/test_runtime_timeline.py`、`tests/test_runtime_tuning.py` 共 **21 passed**。缓存测试验证命中不调用原子读取器、trial 参数变化复用原始缓存并重建窗口、数据/元信息/特征/外部协变量/图变化后的失效和隔离、损坏重建及禁用行为。

## commit2：完成 CRANN 数据语义与新运行时迁移

问题：旧路径缺少原始目标通道，仅验证目标互斥，未接入新训练入口；空间注意力固定为 24 步；batch=1 会被无条件 squeeze 破坏维度；网格列维被误当特征维，仅归一化第一列。

修改：新增标准 nn.Module，仅从 X 与显式 future_features 预测，外部未来协变量需通过 known_future_columns 声明；不读取 y，不持有 scaler/logger/device。data.sampling.name=crann 明确指定 temporal_history，空间历史仍由 input_window 决定；共享目标时间切分、训练区间归一化、y_raw 与原始缓存。temporal_channel 显式选择历史通道，dim_x*dim_y 必须覆盖节点且目标为单通道。注意力按实际 output_window 创建，保留 batch 维，隐状态随输入设备与类型初始化；n_layers_tem 实际控制编码与解码层数。旧参考数据路径同时保留 y_raw 并修复整网格归一化。

验证：CRANN 的 4 项测试及迁移表测试 **5 passed**；原版 24 步前向/梯度对照、无标签/batch=1、精确原尺度零掩码、目标时间互斥、真实原子文件短训练、独立评估和两轮连续/中断恢复参数逐项相等。提供 crann.json 与 --model CRANN 夹具生成命令。迁移范围更新为 16/100 条记录，84 条仍 pending，未声明全库完成。

## 持续完成的清单

- [x] 基础交通缓存保存、读取、失效和独立目录。
- [x] CRANN 的统一输入、原始目标、完整训练/评估/续训与无标签推理。
- [ ] 全部未延期模型迁移、实际验证与迁移 CSV 更新（基线 15/100 已迁移；模板及第 8 节 Common 事项另行标注）。
- [ ] 扩大后的完整回归、发布示例与最终逐项验收；在此之前不声明主线全部完成。
