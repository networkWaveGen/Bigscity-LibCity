# 阶段五执行记录

实施基线 `3dcae47`；开发分支 `refactor/runtime-phase-5`，交付分支 `refactor/libcity-modernization`。仍保留 13 个运行时入口、15 个已迁移任务—模型条目和 85 个 pending 条目，不在清理阶段把待迁移模型算作已支持。

## commit1：发布当前示例与说明

`fe04202` — docs: publish layered source examples and current runtime usage

五类任务均有随仓库发布的五区块 JSON 模板；示例生成器复制模板并在用户指定目录创建原子数据。小型数据夹具移到不依赖 pytest 的 libcity.testing.fixtures，工具不再从 tests 目录导入代码。中英文 README 和 examples/README.md 提供训练、独立评估、续训与调参命令；hyper_example.json 改为新搜索空间格式。

## commit2：删除已替换运行时，迁移回归

`32aa84e` — refactor: retire legacy execution and migrate regression contracts

删除旧 ConfigParser、Batch/BatchPAD、registry 实验工厂、旧 pipeline/tuning、全部旧 Executor 类、旧命令和历史检查点迁移工具。只有 runtime_checkpoint.py 负责加载，要求 .pt、weights_only=True，且不提供不安全反序列化回退。完整状态必填字段在恢复前检查。

未迁移模型的数学源码保留；旧扁平 JSON 移到不参与安装的 reference/config，四个经典基线的数学函数移到 reference/baselines 并去掉默认 CLI 入口。待迁移数据源码只保留按需编码器查找与 Tensor 字典组批，不重新建立第二套实验入口。可视化工具独立使用明确的数据/输出目录，坐标按 JSON 解析。

旧测试的 55 个函数逐项记录在 [迁移映射](phase5-test-migration.json)。缓存、归一化及模型数值检查保留；旧 API 专用测试由新配置/训练/评估/续训测试覆盖；补充 Tensor 源数据所有权、变长长度、spawn、尾批、旧格式拒绝和重复用户轨迹导出检查。没有把旧测试标为 skip 或虚假通过。

清理后的首轮完整测试 147 项通过；随后补充可视化 CLI 检查，专项 13 项通过。最终数量以干净环境完整验收记录为准，不能与旧接口的 214 项机械比较。删除和参考配置清单见 [phase5-retirement.json](phase5-retirement.json)。

## commit3：干净 CPU、系统只读与实际 CUDA 验收

`b4bcaac` — test: verify clean readonly CPU runs and representative CUDA lifecycles

独立 CPU venv 不继承系统包，通过源码安装 dev/transformer；pip check 通过。用 Git 跟踪文件建立单独源码副本、复制小型原子数据，仅对临时副本设置当前身份的继承拒绝 Write/Delete 权限，原仓库和原数据权限不变。四个新文件/现有文件写入探针均被拒绝；子进程导入路径确认为只读副本。沙箱内 ACL 不生效的尝试没有算通过，最终在原生权限执行环境验证。

在此只读副本中完成 **148 项完整测试，0 失败、0 跳过**；pytest 缓存和临时目录均在独立可写目录。随后从仓库外通过 run_model.py 绝对路径执行训练、独立评估及续训到第三轮，指标一致，权限在结束后恢复。记录见 [只读验收](phase5-readonly-validation.json)。

另一个基础环境未安装 TensorBoard/einops/Ray/Hyperopt/DGL/TorchText，五类任务的 19 条 CLI 命令全部通过，见 [CLI 记录](phase5-cli-validation.json)。当前源码重新执行真实 METR-LA 前七天、207 节点短训练与独立评估，见 [实数据记录](phase5-metr-la-validation.json)；该记录不宣称全量精度基准。

发现机器有现成 CUDA PyTorch 后，建立只继承该 CUDA 栈的独立 venv，不改动原环境。RTX 5060 上 GRU、STGCN、DCRNN 的训练、独立评估和续训均通过，前向钩子确认输入、参数和图 buffer 均在 CUDA；GRU 使用 CUDA float16 AMP，续训参数、计数和 GradScaler 与连续训练对照一致。见 [GPU 记录](phase5-cuda-validation.json)。Windows 没有 MPS，本轮未作 MPS 认证。

## commit4：最终文档交付与验收

本执行记录所在最终交付提交 — docs: complete phase five acceptance and retire obsolete guidance

方案阶段五的三项已打钩，后续讨论未实施。当前源码入口、安装依赖、模型迁移清单和参考材料角色均更新；已删除源码的历史链接改指向删除前提交，便于追溯。完整逐项验收见 [phase5_acceptance_zh.md](phase5_acceptance_zh.md)，机器汇总见 [phase5-acceptance.json](phase5-acceptance.json)。

最终源码安装元数据不再提供失效的 tune 可选组；基础运行和有限搜索不需要 Ray/Hyperopt。已实际执行发布的 hyper_example.json 四组候选，最终只评估一次测试集。更新后的 100 项清单仍明确 85 项 pending，没有把“旧源码保留”当成“可通过新入口运行”。
