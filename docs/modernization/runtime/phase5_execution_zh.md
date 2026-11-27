# 阶段五执行记录

实施基线 `3dcae47`；开发分支 `refactor/runtime-phase-5`，交付分支 `refactor/libcity-modernization`。仍保留 13 个运行时入口、15 个已迁移任务—模型条目和 85 个 pending 条目，不在清理阶段把待迁移模型算作已支持。

## commit1：发布当前示例与说明

`fe04202` — docs: publish layered source examples and current runtime usage

五类任务均有随仓库发布的五区块 JSON 模板；示例生成器复制模板并在用户指定目录创建原子数据。小型数据夹具移到不依赖 pytest 的 libcity.testing.fixtures，工具不再从 tests 目录导入代码。中英文 README 和 examples/README.md 提供训练、独立评估、续训与调参命令；hyper_example.json 改为新搜索空间格式。

## commit2：删除已替换运行时，迁移回归

删除旧 ConfigParser、Batch/BatchPAD、registry 实验工厂、旧 pipeline/tuning、全部旧 Executor 类、旧命令和历史检查点迁移工具。只有 runtime_checkpoint.py 负责加载，要求 .pt、weights_only=True，且不提供不安全反序列化回退。完整状态必填字段在恢复前检查。

未迁移模型的数学源码保留；旧扁平 JSON 移到不参与安装的 reference/config，四个经典基线的数学函数移到 reference/baselines 并去掉默认 CLI 入口。待迁移数据源码只保留按需编码器查找与 Tensor 字典组批，不重新建立第二套实验入口。可视化工具独立使用明确的数据/输出目录，坐标按 JSON 解析。

旧测试的 55 个函数逐项记录在 [迁移映射](phase5-test-migration.json)。缓存、归一化及模型数值检查保留；旧 API 专用测试由新配置/训练/评估/续训测试覆盖；补充 Tensor 源数据所有权、变长长度、spawn、尾批、旧格式拒绝和重复用户轨迹导出检查。没有把旧测试标为 skip 或虚假通过。

清理后的首轮完整测试 147 项通过；随后补充可视化 CLI 检查，专项 13 项通过。最终数量以干净环境完整验收记录为准，不能与旧接口的 214 项机械比较。删除和参考配置清单见 [phase5-retirement.json](phase5-retirement.json)。

## 待完成的验收

独立 CPU 环境已安装源码及 dev/transformer 依赖；继续执行系统权限限制下的只读源码/数据、完整回归与端到端生命周期，随后更新方案勾选和最终报告。
