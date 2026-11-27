# 阶段五验收报告

基线 `3dcae47`，交付分支 `refactor/libcity-modernization`。阶段五三项要求均已完成：切换随仓库发布的用法并清理旧实现、干净 CPU 与系统只读验收、交付可运行示例和完整证据。本轮未扩展待迁移模型范围：仍为 13 个入口、15 条已迁移任务—模型记录，85 条 pending。

| 原方案要求 | 实际完成与验收证据 | 结果 |
| --- | --- | --- |
| 迁移脚本、示例、测试；删除旧入口、ConfigParser、Batch/BatchPAD、Executor、旧检查点加载；更新文档 | 五类 [示例与生命周期](../../../examples/README.md)；新 README 与 [入口说明](../entrypoints.md)；删除旧运行时、Ray/Hyperopt 命令与 tune 安装别名；旧 JSON 移到不安装的 reference；[删除清单](phase5-retirement.json)；新测试断言模块/Executor 类不存在，拒绝历史格式，验证 Tensor 字典、源数据所有权、padding/lengths、spawn、尾批和重复用户导出 | 通过 |
| 干净 CPU 全部回归及端到端；仓库外绝对路径、只读源码/数据与可写缓存/运行目录；有 GPU 时补验证 | 独立 Python 3.12.13 venv，不继承系统包；源码安装 dev/transformer，pip check 通过。临时源码/数据副本设置实际继承拒绝 Write/Delete，四个新建/修改文件探针均被拒绝，确认导入来自只读副本；在其中完整测试 **148 passed，0 failed，0 skipped**，随后仓库外训练/评估/续训到第三轮一致。[只读机器记录](phase5-readonly-validation.json)。RTX 5060 上 GRU/STGCN/DCRNN 实际 CUDA 生命周期均通过，GRU AMP 恢复与连续训练对照一致，[GPU 记录](phase5-cuda-validation.json) | 通过 |
| 交付训练、续训、独立评估示例、完整迁移表、实数据短跑和测试报告 | 五类分层 JSON 模板与生成器；无 TensorBoard/einops/Ray/Hyperopt/DGL/TorchText 的基础环境中，**19 条实际 CLI 命令通过**，[记录](phase5-cli-validation.json)；发布的 hyper_example.json **4 个 trial，仅最终测试一次**，[记录](phase5-published-search.json)；完整 [100 条迁移 CSV](model-migration.csv)；当前代码重跑真实 METR-LA 前七天、207 节点、12→12 步一轮训练与独立评估一致，[记录](phase5-metr-la-validation.json) | 通过 |

## 旧测试如何处理

阶段三、四的 214 项与当前的 148 项不是同一套接口用例，不能只比较数字。旧 ConfigParser、Executor、Batch 和历史格式迁移 API 已删除，其专用测试不再保留为假兼容层。旧 55 个测试函数均在 [逐项映射](phase5-test-migration.json) 中列明去向：数值、缓存与归一化回归保留；公共训练、全局指标、选模、续训与配置行为由当前运行时用例覆盖；增加了退休接口不复活、旧格式明确拒绝及新的组批/导出检查。没有 skip、xfail 或用空断言替代旧行为。

## 运行环境与边界

- 干净 CPU 环境：Windows、Python 3.12.13、PyTorch 2.15.0+cpu、NumPy 2.5.5；完整测试实际从只读副本执行。源码/数据权限只在临时副本设置，原仓库和原数据权限未修改，测试结束后恢复副本权限。
- GPU 环境：独立 venv 复用已有 PyTorch 2.10.0+cu128 / CUDA 12.8，Python 3.11.1、NumPy 2.4.5，RTX 5060 Laptop GPU。它是代表模型 GPU 验证，不是干净 CPU 环境认证；其他模型组合和 MPS 未认证。
- METR-LA 记录是前七天实数据功能短跑，未宣称四个月全量精度基准。LINE 的持出边指标不等同于下游分类精度，DeepTTE 使用已知计划路线几何输入；模型限制见 [迁移表说明](models_zh.md)。
- 85 条 pending 模型的数学源码和旧参考配置仍保留，但没有第二套旧执行入口。37 个特殊 Dataset 深度重构、Common 合并、外部调参后端、分发形式、DDP 与 LLM 等方案后续讨论事项未开展。

提交顺序、修改说明与复验方法见 [执行记录](phase5_execution_zh.md) 和 [当前使用指南](README_zh.md)。全部为新提交，不改写历史，不添加 co-author，不做文件哈希校验。
