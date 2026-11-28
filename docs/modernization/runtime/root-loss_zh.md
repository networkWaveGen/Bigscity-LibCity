# RMSE 的累积更新与评估聚合

本批是 CCRNN 迁移所需的公共训练能力，不代表 CCRNN 已完成接入。当前通过 Python 的自定义 loss_fn 组合使用；模型的命令行配置将在其迁移批次单独开放。本批不改变已接入模型的默认损失，也不把 MSE 改称为 RMSE。

## 问题与目标

CCRNN 原 calculate_loss 使用原尺度 masked_rmse。RMSE 对整组有效残差定义为 sqrt(sum(error²)/count)。这个开方是非线性的，直接平均各批 RMSE 得不到整体验证 RMSE；将各微批 RMSE 独立反向再平均，也得不到合并有效批次的 RMSE 梯度。有效目标数量不同、存在缺失/掩码和最后一个不足累积组时，差异尤其明显。

因此，新接入需要在累积和评估时保存可相加的平方误差统计，最后再执行开方。这里只保留标量累计值，不保留多个微批的计算图。

## 实现

- RootMeanSquaredLoss 复用公共原尺度变换、float64 保留、半精度提升、有限目标和原始零值掩码。它的 StepResult.loss 明确表示批次平均平方误差统计，weight 表示有效目标数；这两个字段在本损失中不是已经开方的逐批目标值。
- 每个微批对 loss*weight 反向，即累计平方误差和的梯度。累积组结束时，将梯度除以 2*count*sqrt(sum(error²)/count)，等价于对完整有效批次的 RMSE 反向。归一化在 AMP unscale 之后、梯度裁剪与优化器更新之前执行，Adam/SGD、步进调度等仍由公共组件管理。
- 全部有效残差恰好为零时，选择有限的零次梯度，使用 count 作为非零除数；优化器的动量、权重衰减和正常步进规则继续生效。完全没有有效目标时除数为零，仍沿用公共组件跳过参数更新与 step 调度的规则。
- aggregate_loss 调用损失对象可选的 aggregate(total,weight)，在验证/独立评估和 epoch 训练日志中统一完成开方。未声明此方法的现有任务仍使用原来的加权平均。
- 默认 train_loop 调用损失对象可选的 gradient_divisor(total,weight)。没有该方法的损失仍按原有效目标数归一化。该扩展用于训练步返回的可加统计，不能把额外辅助目标随意混入平方误差统计再宣称是同一个 RMSE。

RootMeanSquaredLoss 仅接受显式的 rmse/masked_rmse 名称。rmse 保留所有有限目标；masked_rmse 还按 mask_value 排除原尺度目标，mask_value=null 则只排除缺失值。空集合的汇总返回 0，与公共评估约定一致。最佳模型与早停接收的是全局 RMSE，而不是平均逐批 RMSE。

## 验证证据

新增 16 项专项检查包括：

- 四组掩码/非掩码及原尺度 float64 统计和梯度，与独立直接 RMSE 计算对照。
- 精确拟合的有限零梯度，以及全掩码/缺失目标的可微零统计。
- 四种评估 batch_size，包含缺失值、原始零值和尾批，loss 均与全量有效残差的直接 RMSE 一致，预测导出保持原样。
- 有/无梯度裁剪两种累积更新，与逐次直接整批 RMSE 反向、SGD 动量、权重衰减和 step 调度逐项对照；包含最后一个不足累积组。
- 双层 GRU dropout、shuffle、Adam、metric 调度及完整随机状态的连续/中断恢复完全一致。
- 全空累积组不更新参数和 step 调度，真实 CPU AMP 与短累积尾组可运行。

完整回归与逐项结果见 [汇总报告](remaining-validation.json)。本批保持模型清单 68/100；CCRNN 的训练区间 SVD 图适配、原生模型与 CLI 生命周期仍在后续迁移中。
