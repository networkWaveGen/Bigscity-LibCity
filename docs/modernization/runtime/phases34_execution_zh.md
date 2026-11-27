# 阶段三、四实施记录

基线为 `41a334d`，开发分支 `refactor/runtime-phases-3-4`，交付分支 `refactor/libcity-modernization`。阶段五及方案第八节继续延后。全部为新提交，不添加 co-author，不做文件哈希校验。

## 第一批：预测输入与目标时间轴

- 用目标窗口完整所属的时间区间选择样本；跨训练/验证/测试边界的窗口被丢弃，历史可以使用切分点之前的观测。
- 按原始时间点拟合训练区间归一化器，不用重复滑窗或验证/测试值拟合；支持比例与显式 train_end/val_end 时间边界。
- X 为历史观测；future_features 只含日历和明确声明的已知未来外部协变量；y 只含归一化后的目标；y_raw 保留原始目标供指标与掩码使用。训练期 Teacher Forcing 仍显式使用 y。
- GRU 滚动预测对不可提前获取的辅助观测沿用最后历史值，不读取未来观测；Trafformer 时间编码只选择已知特征；公共评估、DCRNN、ASTGNN、MTGNN 均使用原始目标做掩码。
- CRANN 四组输入/标签同步按目标窗口划分，并隔离旧缓存命名。通用适配不再读写旧滑窗展开缓存；cache 参数保留目录配置，深度缓存改造仍在后续事项内。
- 同权重计算对照继续保留；旧 Trafformer 参考测试在测试端重组旧 y 格式，生产入口不恢复对标签的依赖。

验收：tests/test_runtime_timeline.py 覆盖目标时间互斥、历史越界合法、验证集污染不改变训练统计、无标签/标签污染推理、观察量隔离、显式未来协变量、原始零值掩码与 CRANN 四字段对齐。真实 METR-LA 短跑由 tools/validate_runtime_real_data.py 记录。

阶段三完整回归 **190 项通过**，记录见 [phase3-validation.json](phase3-validation.json)。真实 METR-LA 前七天、207 节点、12→12 步 GRU 单轮训练及独立评估一致，记录见 [metr-la-phase3.json](metr-la-phase3.json)；该记录是功能短跑，不是全数据精度基准。

## 后续批次

继续完成五类任务代表组合、标准与定制模型批次、全模型审计状态表、统一调参及最终验收。此记录不将尚未执行的内容标为完成。
