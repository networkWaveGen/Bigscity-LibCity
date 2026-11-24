# LibCity 现代化改造实施方案的执行

下文保留重排前的提交编号与验收记录。现行导入约定和审查进展见[改造说明](README.md)与[审查记录](review-notes_zh.md)。

**提交编号映射（从最早到最晚）。**

以下编号对应本重构分支从固定上游基线之后的 Git 提交顺序，不把官方原有历史计入。例如：**`cf1af99` → test: validate modern core and fix heterogeneous sample padding → commit7**。正文统一用 commit 编号引用。

| 短提交号 | 原始 commit 标题 | 顺序编号 |
| --- | --- | --- |
| [8b68efb](https://github.com/networkWaveGen/Bigscity-LibCity/commit/8b68efbec0d4e17b78f8f9eb2a565d7d56f8feaa) | refactor: lazy-load optional components and isolate tuning | **commit1** |
| [e214b0c](https://github.com/networkWaveGen/Bigscity-LibCity/commit/e214b0c4b82d5d6f34d098ae83e39aa7ef4891bb) | build: separate modern core requirements from optional legacy stacks | **commit2** |
| [49acbd2](https://github.com/networkWaveGen/Bigscity-LibCity/commit/49acbd29179ee7c7cdde8cf8a6517325b7a17668) | fix: support modern NumPy and safe CPU checkpoint loading | **commit3** |
| [f36a7c5](https://github.com/networkWaveGen/Bigscity-LibCity/commit/f36a7c5a9a5e42f93410f9ea5ca72cd7b8e655e0) | refactor: centralize component lookup and preserve initialization errors | **commit4** |
| [76053df](https://github.com/networkWaveGen/Bigscity-LibCity/commit/76053df40d06edd96714f90096c55da8bc5e1bff) | perf: avoid redundant dense batch copies and support spawn workers | **commit5** |
| [7672e36](https://github.com/networkWaveGen/Bigscity-LibCity/commit/7672e36895016c945a6ed8870b00c0f4bf51e174) | perf: remove full prediction copies during traffic evaluation | **commit6** |
| [cf1af99](https://github.com/networkWaveGen/Bigscity-LibCity/commit/cf1af9945f5d450e8ab987cfd2c47db2e1f4f226) | test: validate modern core and fix heterogeneous sample padding | **commit7** |
| [ba1374e](https://github.com/networkWaveGen/Bigscity-LibCity/commit/ba1374e404f666f59944f18dbb6219587f5acc59) | docs: record model compatibility and concentrated validation results | **commit8** |
| [3b57551](https://github.com/networkWaveGen/Bigscity-LibCity/commit/3b57551b4895ac4f1bc7e69d2dc11a7fdfdf261c) | style: keep lazy export declarations readable | **commit9** |
| [0931d40](https://github.com/networkWaveGen/Bigscity-LibCity/commit/0931d406801b221be38312c56da95460718a075c) | docs: map modernization execution to implementation commits | **commit10** |
| [78453fc](https://github.com/networkWaveGen/Bigscity-LibCity/commit/78453fc10454714eeb1dc3b55dfac8c4092494ac) | audit: archive upstream registrations and verify dependency removal | **commit11** |
| [80a990d](https://github.com/networkWaveGen/Bigscity-LibCity/commit/80a990d95d4b4592d910c3de6e5186134a77a204) | fix: restore optimizer checkpoints consistently across execution paths | **commit12** |
| [6be3388](https://github.com/networkWaveGen/Bigscity-LibCity/commit/6be3388299f009e3c32b782b81697a71614b38f9) | test: validate task entrypoints and repair template registration | **commit13** |
| [ad6afd8](https://github.com/networkWaveGen/Bigscity-LibCity/commit/ad6afd8f9b3e382519e5d54ed69d500d94e26eca) | docs: record completed follow-up audits and concentrated validation | **commit14** |
| [58f5d67](https://github.com/networkWaveGen/Bigscity-LibCity/commit/58f5d6785c8941b9154dde59dcc75c6c37f8d205) | docs: update execution checklist and chronological commit map | **commit15** |
| [本文追加记录][commit16] | docs: append detailed change descriptions for each commit | **commit16（本次追加）** |

本文件依据原《LibCity现代化改造实施方案.md》，核对已落地的代码、提交与验证记录，逐项说明具体做了什么。原方案保持不变；本文件保留原有 34 个勾选项，并在每项下面补充执行证据。AgentCity 仍在后续阶段接入。

更新日期：2026 年 11 月 24 日。原来 6 项部分完成内容已逐项补齐，**原清单 34 项均已按注明范围完成**。上一轮新增 commit11 至 commit15，分别处理基线与依赖审计、检查点、入口验收、验证证据、执行文档；本次在末尾追加逐个提交的修改明细，对应 commit16。完成原清单不等于全部旧模型精度、真实 GPU 或历史环境回归已经通过；这些验证边界在第五步明确列出。

**工作对象与范围。** 本地参考代码为 `D:\LibCity 2.0\LibCity的github代码`；实际修改目录为 `D:\LibCity 2.0\LibCity-modernization`。工作分支为 `refactor/libcity-modernization`，用户远程仓库为 [networkWaveGen/Bigscity-LibCity](https://github.com/networkWaveGen/Bigscity-LibCity)。官方上游为 [LibCity/Bigscity-LibCity](https://github.com/LibCity/Bigscity-LibCity)，开发起点见[固定上游基线][上游基线]。

本轮不修改 AgentCity、AgentCity 内嵌的 LibCity，也不以 `LibCity2.0-main` 中的拼接副本作为实验内核。

| 本轮实施 | 暂缓实施 |
| --- | --- |
| 同步上游、固定提交、建立独立分支 | AgentCity 集成、Agent 工作流和网页问题 |
| 核心 Python、PyTorch、NumPy 及必要数值库升级 | TorchText、DGL 等可选依赖的系统性更新或替换 |
| 可选依赖隔离、组件按需加载、清理无实际用途的依赖 | Optuna 引入和调参后端替换 |
| CPU、设备选择、旧 NumPy 调用及明确的路径错误修复 | 全局配置体系、数据目录和产物目录重新设计 |
| 依据实际职责整理模块，减少重复实现与耦合 | 稳定实验结果接口、自动查论文与自动提 PR |
| 确认用途后，优化明确多余的数据复制、转换与评估缓存 | 滑动窗口全面重写、mmap、算法替换、编译加速等探索性优化 |

**执行原则。** 核心实施与执行清单更新对应 commit1 至 commit15，前 10 个为首轮提交，随后分批补齐 5 个；本次修改明细追加为 commit16。每批形成可审查的改动。内部分类可以调整，以职责清楚、冗余更少和依赖范围更小为判断依据。优先保持原有命令入口及常用模型接口；确需调整的接口记录对应关系。

可选依赖只负责隔离安装和导入，不能保证旧组件在现代核心环境下兼容。保留相关模型代码，明确记录所需依赖、兼容状态和后续处理事项。

**首轮 commit1 至 commit9 的具体改动。**

| 提交 | 主题 | 具体文件和改动 | 结果与范围 |
| --- | --- | --- | --- |
| [commit1] | 可选组件延迟加载、隔离调参 | 新增 `libcity/_optional.py`；将各类 `__init__.py` 的集中导入改为导出映射；Ray 调参拆到 `pipeline/tuning.py`，执行器仅在调参时加载 Ray。 | 普通 GRU 实验不会因入口导入而要求安装 Ray、DGL、TorchText 等；旧调参入口仍保留。 |
| [commit2] | 现代安装配置、核心和可选依赖分组 | 新增 `pyproject.toml`、`requirements-legacy.txt`、现代化说明；基础 `requirements.txt` 改为安装当前项目；模型相关依赖分组。 | 可以安装现代核心，再按功能选择 extra；旧依赖保留为参考，旧功能兼容性另行标记。 |
| [commit3] | 修复 NumPy、CPU、配置及检查点兼容问题 | 替换旧 NumPy 类型别名；调整 CUDA 判断顺序；包资源读取默认配置；修正 JSON 文件名；新增公共检查点读取；修正指标返回值处理。 | CPU 启动、现代数值库及公共模型状态加载可用于所测核心路径；并未统一重做全部路径或所有模型检查点。 |
| [commit4] | 统一组件查找、保留真实构造错误 | 共享 `resolve_component`，整理 `get_model`、`get_dataset`、`get_executor`、`get_evaluator` 的查找；构造对象放在查找兜底之外。 | 减少重复查找逻辑；模型初始化的实际错误会直接暴露，避免被误报为组件不存在。 |
| [commit5] | 减少批处理复制、支持 spawn worker | 整理 `Batch` 的 Tensor 转换和 padding；新增顶层 `BatchCollator`；规则数值字段减少深拷贝，变长字段保留必要隔离。 | 减少数据拷贝与转换，并使 collator 能被 Windows 多进程序列化；后续测试补上所有权与异形补齐修复。 |
| [commit6] | 去掉评估阶段额外的全量转换 | `TrafficStateExecutor.evaluate` 改为收集 CPU Tensor 后 `torch.cat`；增加默认开启的 `save_predictions` 配置。 | 指标仍按原全数据语义计算；可关闭预测文件保存，但仍需要全量拼接预测，未实现流式指标。 |
| [commit7] | 集中检查及发现问题的修复 | 新增 `tests/test_modern_core.py`；修复 NumPy 2 下不同形状 X/y 的尾样本补齐、稠密字段所有权及 padding 边界；调整构建工具要求。 | 19 项集中检查通过，覆盖 GRU/STGCN 前向反向、批处理、评估和 GRU 合成数据短训练等。 |
| [commit8] | 记录兼容清单、环境和验收证据 | 新增 `tools/compatibility_report.py`、100 项静态清单、验证记录、环境 JSON、版本约束；更新中英文 README。 | 给出实际测过什么、仅静态清点什么、后续暂缓什么的明确记录；没有宣称 100 个模型全部通过。 |
| [commit9] | 整理延迟导出声明格式 | 整理导出列表和重复声明，保留公开导出名称及顺序，清理新调参模块尾部空行。 | 改善代码可读性；该次提交只做必要语法检查，后续 39 项集中检查已覆盖当前版本。 |

可查看[截至 commit14 相对基线的完整差异][变更对比]。本轮的 commit11 补齐基线/依赖/历史记录审计，commit12 补齐检查点恢复，commit13 完成入口验收，commit14 归档集中结果，commit15 更新此执行文档。新增的基础安装仍是一个公共核心环境，没有为每个模型生成单独的最小依赖环境。可选组解决安装和加载耦合，不等于相应旧模型已经兼容现代版本。

以下保留原方案的 34 项内容逐项核对。**打钩表示已在注明范围内完成；本轮补齐的 6 项以“已补齐”标识，原先未勾选的方框现已打钩。** 创建克隆和建立分支属于 Git 操作，本身不会产生 commit；对应项注明了记录这些操作的文档提交，未把文档提交冒充为操作本身。

**第一步：固定开发基线并建立分支。**

- [x] 取得官方 LibCity 当前主线，记录来源地址、分支和提交 SHA。

  执行记录：已建立官方上游 `https://github.com/LibCity/Bigscity-LibCity.git` 的固定基线：`master`，提交 `5a6391d41944e937f2c15e9be85ab7f40ac8b23e`（2024-12-18）。这是本轮取得的开发快照，不能据此声称以后仍与官方最新主线同步。拉取属于 Git 操作；来源、分支及基线的文档记录见 [commit2]。

- [x] 检查现有本地目录是否具有 Git 历史；必要时新建独立克隆，保留原参考副本。

  执行记录：参考目录没有 Git 历史，另建 `D:\LibCity 2.0\LibCity-modernization` 作为正式工作副本，保留原参考目录。克隆属于 Git 操作，本身没有对应代码 commit；工作副本的首批可审查改动为 [commit1]，基线说明见 [commit2]。

- [x] 创建重构分支，建议名称为 `refactor/libcity-modernization`。

  执行记录：实际工作分支为 `refactor/libcity-modernization`，所有本轮重构提交均在此分支；用户仓库配置为 `origin`，官方仓库为 `upstream`。创建分支不产生 commit，分支记录见 [commit2]。

- [x] 保存改造前的依赖、模型和组件注册清单。

  执行记录：**已补齐。** [commit11] 从固定上游提交直接导出 19 个包、220 个公开组件，并归档原始任务配置；[commit13] 补充 `baseline_models.csv` 的 100 项旧模型及组件映射、`task_registration_changes.json` 的前后对照。当前审计从固定上游提交读取旧依赖，基线清单可通过 [`baseline_audit.py`](audit/baseline_audit.py) 重新生成。

- [x] 记录本机 Python、操作系统、CPU 或 GPU 环境，并保存已有可用的历史实验记录。

  执行记录：**已补齐。** [commit11] 保存已有实验的检索范围和结论，检查原参考副本、`LibCity2.0-main/LibCity` 及工作副本缓存、日志目录，未找到可归档的历史实验产物；[commit14] 保存当前 Python、系统、CPU、依赖版本和实际测试记录。本项完成的是“保存已有可用记录”的核查；没有已有历史精度结果就明确记录不存在，没有补造旧实验，也不声称已通过历史精度回归。

这一阶段只建立来源和比较依据，不要求预先执行全量模型训练。

**第二步：拆开可选依赖和加载链。**

- [x] 追踪入口、模型、Dataset、Executor、Evaluator 及其 `__init__.py` 的实际导入链。

  执行记录：[commit1] 拆开集中导入链，[commit4] 统一组件查找；[commit13] 进一步逐个实际尝试 220 个公开名称、100 项任务配置及其组件、根目录脚本的 LibCity 导入，补齐各任务入口检查。可选依赖缺项与核心错误分别记录，详见[入口验收][入口验收]。

- [x] 将基础实验必须的依赖与 Ray、TorchText、DGL、FastDTW、Gensim、torchdiffeq 等特定功能依赖分开管理。

  执行记录：[commit2] 新建 `pyproject.toml`，划分现代核心与 `tune`、`graph`、`dtw`、`embeddings`、`geosan`、`ode` 等功能组。现行审计直接从上游基线读取原依赖。旧依赖已隔离，未逐个升级。

- [x] 普通训练不直接要求 Ray；启用原有调参功能时才加载相关组件，本轮保留现有调参后端。

  执行记录：[commit1] 把 Ray 调参入口移到 `libcity/pipeline/tuning.py`，执行器仅在启用调参时导入 Ray，保留历史调参入口名称。[commit7] 覆盖不加载 Ray 的核心导入。Ray 旧 API 的实际运行兼容性仍待确认。

- [x] 调整组件创建方式，使选择一个模型时只加载所需模块；原有名称解析与配置映射同时保持有效。

  执行记录：[commit1]、[commit4] 保留原公开名称与工厂查找，[commit9] 整理声明；[commit13] 核对全部旧公开名称及历史直接路径，未删除旧公开名称。当前可用核心组件完成实际导入及对象身份比较，可选组件因明确缺少依赖而隔离记录。

- [x] 在确认无直接或间接用途后，将无关依赖移出基础安装清单。

  执行记录：**已补齐。** [commit11] 对旧 26 项依赖逐项检查全部项目 Python 导入，并遍历已安装现代核心的活动 `Requires-Dist` 传递依赖。TensorBoardX、aioredis、aiohttp、idna 在源码及当前核心依赖闭包中均无用途，移出基础直接依赖；旧 lint 工具归为开发用途，其他依赖对应现代核心或已保留的可选组。没有未分类移除项、没有缺失的核心分发包；可选栈的间接依赖交由其自身声明安装。该结论针对已验证的平台与核心环境，详见[依赖审计][依赖审计]。

- [x] 补充包配置和依赖分组，使基础安装与可选安装方式可复现。

  执行记录：[commit2] 提供包配置及可选分组，[commit7] 修正构建工具要求；[commit11] 显式声明审计工具使用的开发依赖；[commit14] 记录重新构建 wheel 成功、包含全部 174 个 JSON 配置，并在隔离 Python 模式下直接导入 wheel 成功。使用的是已有验证环境中的依赖，未宣称完成全部平台或依赖组合的独立安装。

- [x] 缺少可选依赖时，提示具体模型、缺失组件及安装方式；避免影响不使用该组件的模型。

  执行记录：[commit1] 新增 `libcity/_optional.py`，对已登记的缺失可选模块给出请求组件、缺失模块和 extra 安装提示；[commit2] 补充依赖映射与旧组件兼容性说明。[commit7] 检查基础导入不加载无关组件。该提示覆盖已登记的缺失模块；第三方库自身的 API 或二进制报错仍保留原错误。

阶段完成标准：不安装 Ray、TorchText、DGL 等无关组件时，基础模型所需的模块仍可导入和创建。

**第三步：升级核心环境并修复兼容问题。**

实际验证环境为 Windows 11、Python 3.12.13、CPU PyTorch 2.15.0、NumPy 2.5.5、pandas 3.1.1、SciPy 1.18.1、scikit-learn 1.9.1。最新环境与结果见 [commit14] 的[集中检查记录][集中结果]，原[版本约束][版本约束]保留。CUDA 需要选择相应 PyTorch 构建，但本轮没有真实 CUDA 验证。

- [x] 修改基础依赖约束，建立干净的现代环境，核查安装解析结果。

  执行记录：[commit2] 建立现代版本约束，主环境为 Python 3.12；[commit8] 保存已验证的约束文件；[commit14] 记录本轮实际环境和 41 个已安装包的元数据兼容检查通过。验证为 Windows CPU 环境，未据此认证整个版本区间或 CUDA 组合。

- [x] 替换 `np.float`、`np.int` 等旧调用，保持原有类型语义，避免无意改变精度和索引类型。

  执行记录：[commit3] 替换数据集、Node2Vec、GWNET、HIEST 等源码中的旧 `np.float`、`np.int` 别名，按原别名含义改用 `float`、`int`，没有给 NumPy 增加全局猴子补丁；[commit7] 同时修复 NumPy 2 对异形样本补齐的处理。

- [x] 修复先调用 CUDA 再判断可用性的顺序，明确 CPU 与指定 GPU 的行为。

  执行记录：[commit3] 先判断 GPU 请求及 `torch.cuda.is_available()`，再调用 `set_device`，无 CUDA 时使用 CPU；[commit7] 检查禁用 GPU 和 CUDA 不可用两个分支。真实 GPU 上的设备选择与运行尚未验收。

- [x] 检查受升级影响的 PyTorch、NumPy 及数值库接口；对实际阻塞本轮核心流程的调用进行必要适配。

  执行记录：[commit3] 适配 NumPy 旧类型别名、scikit-learn 指标返回标量和 PyTorch 检查点加载；[commit7] 修复异形数组补齐，完成核心前向、反向及短训练检查。范围限于本轮核心路径，不代表每个旧模型的全部接口均已适配。

- [x] 检查模型和优化器检查点的保存、加载及设备映射；需要迁移的格式单独记录。

  执行记录：**已补齐。** [commit12] 将库内全部 `torch.load` 集中到公共函数，统一交通/轨迹恢复、按 epoch 恢复、PDFormer 初始化、GeoSAN 加载及调参结果读取的设备映射；支持模型状态字典、模型/优化器二元组及 epoch 字典，新增显式旧对象迁移工具。10 项检查包含真实更新后的非空 Adam 状态恢复，下一步损失、模型参数和优化器动量与不中断训练完全一致；设备转发、模型权重单独加载、旧格式迁移也已覆盖。实际硬件验证为 CPU；CUDA 只检查参数转发，未知旧类和未保存的调度器/随机状态的限制见[检查点说明][检查点说明]，结果见 [commit14]。

- [x] 对可选依赖或旧组件产生的兼容问题进行隔离、标记，不扩展为本轮全面替换任务。

  执行记录：[commit2] 隔离旧功能栈；[commit13] 将实际检查结果写入兼容清单：82 项导入通过、16 项缺少已登记可选依赖、2 项开发模板。旧可选栈保留并明确标记，没有扩大为 TorchText、DGL 或旧 Ray API 的全面迁移。

- [x] 仅修复明确导致启动或加载失败的路径问题，保留全局路径体系重做为后续事项。

  执行记录：[commit3] 用 `importlib.resources` 读取包内默认 JSON，支持显式 `.json` 配置文件名并以 UTF-8 读取；[commit7] 检查从其他启动目录读取默认配置及指定配置。`raw_data`、缓存和产物目录的整体路径设计仍沿用现有方式。

最终替换源码中的旧用法；已有临时兼容补丁若仍有用途，应限定范围并记录原因。

**第四步：集中整理结构并修复数据和评估问题。**

模块调整以真实调用关系为依据，不预先强制维持五模块分类，也不要求把所有任务塞进同一种执行器。合并重复的公共功能，保留交通状态预测、轨迹预测、ETA、传统地图匹配等任务必要的差异。

- [x] 梳理组件创建、设备处理、重复工具函数和训练公共逻辑，确定一批职责明确的调整。

  执行记录：[commit4] 抽取公共组件查找，消除多层兜底查找吞掉构造错误的问题；[commit1] 隔离调参公共入口，[commit3] 统一两类公共执行器的检查点读取。没有进行所有执行器训练循环的全面合并，后者列为后续工作。

- [x] 检查迁移后的导入路径和注册映射，防止文件移动后旧入口失效。

  执行记录：**已补齐。** [commit13] 对全部 220 个旧公开组件尝试当前公开路径与历史直接路径；200 项导入和身份比较通过，20 项明确缺少可选依赖。100 项任务配置及其组件解析全部检查，84 项通过、16 项明确缺少可选依赖；脚本导入 19 项通过、3 项需调参依赖，两个基础 CLI 的 `--help` 通过，无未分类核心导入错误。修复上游 TemplateTLP 缺少任务映射；模板仍按开发模板标记。完整逐项结果见[入口验收][入口验收]。

- [x] 检查批处理深拷贝的用途；在确认原始样本不会被修改后，减少不必要的复制。

  执行记录：[commit5] 对规则数值字段减少逐样本深拷贝，变长或不透明字段保留必要复制，padding 使用自有列表；[commit7] 补齐稠密字段在 Tensor 转换前后的样本所有权检查与修复，避免回写原始数据。

- [x] 减少可证明重复的 List、NumPy、Tensor 转换，并保持字段、dtype、形状、padding、mask 和样本顺序。

  执行记录：[commit5] 使用 NumPy 汇整及 `torch.as_tensor` 减少冗余转换，已有 Tensor 直接调整设备/类型，新增可被 Windows spawn 序列化的 `BatchCollator`；[commit7] 覆盖值、dtype、padding、长度、不完整尾批、字段异常和单 worker。没有重写所有模型专用 mask。

- [x] 检查评估阶段保存全部预测的必要性；对能够保持原指标语义的部分减少缓存或复制。

  执行记录：[commit6] 评估时收集 CPU Tensor 并用 `torch.cat` 合并，去掉全量预测转 NumPy 后再转回 Tensor 的额外副本。为保持全局指标语义，仍会保留并拼接全部预测，没有实现流式评估；[commit7] 检查修改后的结果。

- [x] 保留必要的预测保存能力，核对样本权重、缺失值掩码及各预测步的统计方式。

  执行记录：[commit6] 新增 `save_predictions`，默认仍保存预测归档，可设为 `false` 关闭；[commit7] 对零标签、不齐尾批、10 项指标、逐步及累计平均模式和归档值进行检查。比较基准是当前评估器对全量拼接数据的计算，不是历史环境的真实数据精度结果。

- [x] 其他可能的优化只登记位置、原因和后续验证方法，暂不修改。

  执行记录：[commit8] 开始登记暂缓工作，[commit14] 更新记录：流式指标、进一步执行器去重、资源生命周期、全局路径、窗口采样、编译/分布式等仍暂缓，没有为把清单打钩而扩展为全面重写。

已对本轮批处理改动进行同环境微基准比较，结果见第五步。尚未测量整体训练加速、GPU 显存或真实数据集端到端性能，不能将该微基准结果外推到全部实验。

**第五步：大体改好后集中验证。**

主体实现完成后集中运行整个 `tests` 目录，**39 项通过，18.82 秒，1 条 warning，无失败、错误或跳过**。该次运行包含原核心 19 项、检查点 10 项、跨任务 10 项；参数化组合计入测试项数，不是 39 个模型。代码截至 commit13，结果与环境归档于 [commit14]。测试前完成入口及依赖审计，格式和文档更新后没有重复跑训练。

| 检查项 | 实际执行情况 | 结论边界 |
| --- | --- | --- |
| 基线与依赖 | 19 包/220 公开组件、100 项旧模型清单及前后映射；26 项旧依赖的源码和传递依赖审计；41 个安装包元数据兼容检查通过 | 依赖闭包对应当前平台与现代核心，不是所有旧可选栈的运行认证 |
| 公开组件及脚本 | 220 项逐个尝试，200 项通过、20 项可选缺失；脚本导入 19 项通过、3 项调参依赖缺失；两个基础 CLI 帮助入口通过 | 可选组件缺失已明确记录，不冒充导入或训练通过 |
| 任务注册 | 100 项实际配置与组件解析，84 项通过、16 项可选缺失；无未分类核心错误 | 其中含 2 个开发模板，不作为已实现算法统计 |
| 各任务工厂 | 交通 GRU、轨迹 GRU、DeepTTE、HMMM、LINE 的真实模型/执行器/评估器；轨迹及 ETA 数据集/编码器 | 不等于每个模型构造和所有专用数据处理都已运行 |
| 前向、反向和短训练 | 原交通 GRU/STGCN；新增轨迹 GRU、DeepTTE、LINE 的有限损失/梯度；保留 GRU 合成原子数据的 1 epoch 全管线训练 | HMMM 只检查传统模型工厂；没有真实基准精度对照 |
| 模型/优化器恢复 | 交通、轨迹、epoch 三条真实执行器保存/恢复入口；非空 Adam 状态后继续一步，与未中断结果一致；旧格式迁移及仅模型权重处理 | 实测为 CPU；CUDA 参数转发检查不替代 GPU 训练；旧格式缺失的调度器/RNG 状态不会自动补齐 |
| 数据与评估 | 保留值、dtype、所有权、padding、变长字段、尾批、spawn worker、10 项指标和预测归档检查 | 评估以当前评估器的全量结果为对照，仍需全量拼接预测 |
| 安装与打包 | 重新构建 wheel，包含 174 个 JSON，验证新模板映射，隔离 Python 模式下直接从 wheel 导入成功 | 使用现有已验证依赖环境，没有另建全依赖安装环境 |

机器记录含 39 个逐项测试名称、所测代码提交和代表模型的验证程度，见[集中检查记录][集中结果]；完整说明见[验证记录][验证记录]、[入口验收][入口验收]和[检查点说明][检查点说明]。

**批处理微基准沿用首轮记录。** 同一现代环境比较上游与新批处理：64 个 float32 样本，`X=(12,128,2)`、`y=(3,128,1)`，CPU 单 Torch 线程，预热 3 次、20 次中位数，计时前逐元素核对输出。本轮没有修改该性能实现，因此没有重复计时。

| 测量 | 上游实现 | 修改后 |
| --- | ---: | ---: |
| 批处理及 Tensor 转换中位耗时 | 0.227 ms | 0.057 ms |
| tracemalloc 记录的峰值分配 | 1,693,296 字节 | 888,064 字节 |

该数据不能外推为整体训练加速；tracemalloc 不包含 PyTorch 原生内存或 GPU 显存。

**仍未进行的额外验收。** 没有找到已有历史实验结果，因此不能进行历史环境精度比较；真实交通基准精度回归、全部模型训练认证、真实 CUDA/Linux 运行、未知自定义旧对象迁移、整体训练性能对比，均未在本轮获得验证。原方案约定的可选栈隔离、代表模型验收和限制记录已完成，未将上述未验证事项伪装为通过。

**兼容状态记录。**

[commit13] 将[模型兼容清单][模型清单]从全体仅静态状态更新为实际配置及导入结果：

| 字段 | 本轮记录 |
| --- | --- |
| `task`、`model`、`module` | 100 项注册，区分同名模型所属任务 |
| `status` | 82 项 `import_passed`、16 项 `optional_dependency_missing`、2 项 `template_only` |
| `validation` | 84 项 `config_and_import_passed`；16 项 `config_passed_import_attempted` |
| `optional_extras`、`issues` | 实际所需可选组、缺失组件及模板说明 |

TemplateTLP 的任务映射已补齐；TemplateTLP 与 TemplateTSP 仍为开发模板。模型的实际创建、前向、反向及短训练程度单独记录在[集中检查记录][集中结果]，不能把导入通过当作精度通过。全部公开导入、任务注册和脚本导入的逐项结果均保存在[入口审计汇总][入口汇总]及其同目录 CSV。

**交付与完成标准。**

- [x] 独立重构分支、固定上游提交和按主题整理的改动记录。

  执行记录：固定上游、独立工作副本及重构分支均已建立；commit1 至 commit14 已按主题保留，执行清单更新对应 commit15，本次末尾修改明细追加为 commit16。最前面的映射按从早到晚编号，与 Git 历史顺序一致。

- [x] 基础安装配置、可选依赖分组和明确的环境版本记录。

  执行记录：[commit2] 提供现代核心及可选分组，[commit8] 保存初始版本约束，[commit11] 补齐依赖审计，[commit14] 保存最新环境、集中检查和构建记录。

- [x] 基础模型运行不被无关可选依赖阻塞。

  执行记录：[commit1]、[commit2] 实现隔离；[commit7] 完成原核心检查；[commit13] 增加所有任务的入口核查和代表工厂检查，[commit14] 记录 39 项集中检查通过。旧可选依赖缺失不阻塞所测基础路径。

- [x] 核心升级引起的明确接口和设备问题已处理。

  执行记录：[commit3]、[commit7] 处理本轮核心版本与设备问题；[commit12] 补齐公共及模型专用加载的检查点适配；[commit13] 修复模板映射。真实 GPU 及旧可选模型完整训练仍不在已验证范围内。

- [x] 本轮结构调整完成，关键入口、导入与组件映射有效。

  执行记录：**已补齐。** [commit13] 完成所有旧公开入口、全部任务注册和根脚本导入的逐项核查，并创建交通预测 GRU、轨迹 GRU、ETA DeepTTE、地图匹配 HMMM、路网表示 LINE 的真实模型/执行器/评估器，另检查轨迹与 ETA 数据集/编码器。所有未安装的可选栈明确归类，无未分类核心错误；[commit14] 记录 39 项集中检查通过。本项验收的是结构与入口，不宣称全部模型均已训练或通过精度回归。

- [x] 已修改的数据处理和评估逻辑保持预期行为，验证程度有记录。

  执行记录：[commit5]、[commit6] 的数据和评估改动已由原核心检查覆盖；[commit13] 增加跨任务检查，[commit14] 记录 39 项通过和原批处理微基准。评估仍保留全量拼接，未声称实现流式评估。

- [x] 可选模型兼容状态表、集中验收记录和暂缓优化清单。

  执行记录：[commit8] 提供初始清单；[commit13] 将清单从仅静态登记更新为逐项配置及导入结果，模板与可选缺项分开标注；[commit14] 保存逐个测试案例、代表模型的实际验证程度和暂缓事项。导入通过仍不等于精度已验证。

- [x] 形成可审查的代码差异；通用修复按主题整理，便于后续提交 PR。

  执行记录：commit1 至 commit14 均按主题保留，执行清单更新作为 commit15 单独提交，本次追加说明作为 commit16。没有添加 co-author，没有进行文件哈希校验。此项表示形成了可逐批 review 的改动，不表示已经向官方主仓库提交或合并 PR。

**原 6 项部分完成内容的闭环。**

| 原缺口 | 本轮完成内容 | 提交 |
| --- | --- | --- |
| 改造前注册清单 | 明确导出旧公开组件、100 项模型与任务配置，保存前后变化 | [commit11]、[commit13] |
| 已有历史实验记录 | 完成指定目录检索并归档“未发现已有产物”；保存当前环境和本轮记录 | [commit11]、[commit14] |
| 移出依赖的间接用途 | 检查全部源码导入、现代核心的活动传递依赖和已安装包兼容性 | [commit11]、[commit14] |
| 完整模型/优化器检查点 | 统一所有加载点，真实非空 Adam 恢复及下一步一致性检查，提供旧格式迁移 | [commit12]、[commit14] |
| 旧入口与注册映射 | 逐项检查全部公开名称、任务配置及脚本导入，修复模板映射 | [commit13] |
| 结构调整验收 | 各任务代表工厂、跨任务前向反向和完整 39 项集中检查 | [commit13]、[commit14] |

后续按原约定继续暂缓：AgentCity 及其内嵌 LibCity、网页与 Agent 工作流、旧可选栈系统性更新、调参后端替换、全局配置与路径重做、稳定结果接口、自动查论文和自动 PR、窗口采样重写及其他探索性优化。

**证据文件。** [基线/历史/依赖审计][审计说明]、[入口验收][入口验收]、[检查点说明][检查点说明]、[验证记录][验证记录]、[集中检查记录][集中结果]、[模型清单][模型清单]。证据链接固定到 commit14；执行清单更新见 [commit15]，本次逐项修改明细见 [commit16]。

[commit1]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/8b68efbec0d4e17b78f8f9eb2a565d7d56f8feaa
[commit2]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/e214b0c4b82d5d6f34d098ae83e39aa7ef4891bb
[commit3]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/49acbd29179ee7c7cdde8cf8a6517325b7a17668
[commit4]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/f36a7c5a9a5e42f93410f9ea5ca72cd7b8e655e0
[commit5]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/76053df40d06edd96714f90096c55da8bc5e1bff
[commit6]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/7672e36895016c945a6ed8870b00c0f4bf51e174
[commit7]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/cf1af9945f5d450e8ab987cfd2c47db2e1f4f226
[commit8]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/ba1374e404f666f59944f18dbb6219587f5acc59
[commit9]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/3b57551b4895ac4f1bc7e69d2dc11a7fdfdf261c
[上游基线]: https://github.com/LibCity/Bigscity-LibCity/commit/5a6391d41944e937f2c15e9be85ab7f40ac8b23e
[变更对比]: https://github.com/networkWaveGen/Bigscity-LibCity/compare/5a6391d41944e937f2c15e9be85ab7f40ac8b23e...ad6afd8f9b3e382519e5d54ed69d500d94e26eca
[现代化说明]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/modernization.md
[验证记录]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/validation.md
[模型清单]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/model_compatibility.csv
[环境版本]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/validation_environment.json
[版本约束]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/constraints/validated-py312.txt
[核心测试]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/tests/test_modern_core.py
[清单工具]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/tools/compatibility_report.py
[commit10]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/0931d406801b221be38312c56da95460718a075c
[commit11]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/78453fc10454714eeb1dc3b55dfac8c4092494ac
[commit12]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/80a990d95d4b4592d910c3de6e5186134a77a204
[commit13]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/6be3388299f009e3c32b782b81697a71614b38f9
[commit14]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/ad6afd8f9b3e382519e5d54ed69d500d94e26eca
[commit15]: https://github.com/networkWaveGen/Bigscity-LibCity/commit/58f5d6785c8941b9154dde59dcc75c6c37f8d205
[审计说明]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/audit/README.md
[依赖审计]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/audit/dependency_audit.json
[入口验收]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/entrypoints.md
[入口汇总]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/audit/entrypoint_audit_summary.json
[检查点说明]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/checkpoints.md
[集中结果]: https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/audit/validation_results.json
[commit16]: https://github.com/networkWaveGen/Bigscity-LibCity/commits/refactor%2Flibcity-modernization/docs/modernization_execution_zh.md

---

## 每个 commit 对应的修改内容

以下与最前面的编号一一对应，按从最早到最晚排列；每节描述该次提交实际带来的改动，后续补齐内容记录在对应的后续提交中。点击提交编号可查看完整 diff，主要文件链接固定到该次提交。

### commit1：按需加载组件，隔离 Ray 调参

**原始标题：** [refactor: lazy-load optional components and isolate tuning][commit1]。

**主要文件：** [公共延迟加载工具](https://github.com/networkWaveGen/Bigscity-LibCity/blob/8b68efbec0d4e17b78f8f9eb2a565d7d56f8feaa/libcity/_optional.py)、[普通实验管线](https://github.com/networkWaveGen/Bigscity-LibCity/blob/8b68efbec0d4e17b78f8f9eb2a565d7d56f8feaa/libcity/pipeline/pipeline.py)、[调参管线](https://github.com/networkWaveGen/Bigscity-LibCity/blob/8b68efbec0d4e17b78f8f9eb2a565d7d56f8feaa/libcity/pipeline/tuning.py)、[交通预测执行器](https://github.com/networkWaveGen/Bigscity-LibCity/blob/8b68efbec0d4e17b78f8f9eb2a565d7d56f8feaa/libcity/executor/traffic_state_executor.py)。

- 新增 `install_lazy_exports` 和可选依赖导入提示；把配置、数据集、编码器、模型、执行器、评估器等 19 个包的集中导入改成按名称延迟加载，保留原公开名称。
- 将调参入口和搜索空间解析移入独立调参模块；保留从历史管线模块导入相关函数的方式。
- 将 9 类执行器的 Ray 导入移到实际启用调参的分支，普通训练入口不再在启动时加载 Ray。

**作用：** 基础模型启动时只触发所选组件的导入，减少无关可选依赖阻塞；旧 Ray API 仍保留，其现代版本兼容性未在该提交中重写。

### commit2：建立现代安装配置和可选依赖分组

**原始标题：** [build: separate modern core requirements from optional legacy stacks][commit2]。

**主要文件：** [标准包配置](https://github.com/networkWaveGen/Bigscity-LibCity/blob/e214b0c4b82d5d6f34d098ae83e39aa7ef4891bb/pyproject.toml)、[基础安装入口](https://github.com/networkWaveGen/Bigscity-LibCity/blob/e214b0c4b82d5d6f34d098ae83e39aa7ef4891bb/requirements.txt)、[原依赖清单](https://github.com/networkWaveGen/Bigscity-LibCity/blob/e214b0c4b82d5d6f34d098ae83e39aa7ef4891bb/requirements-legacy.txt)、[现代化说明](https://github.com/networkWaveGen/Bigscity-LibCity/blob/e214b0c4b82d5d6f34d098ae83e39aa7ef4891bb/docs/modernization.md)。

- 新增包元数据、Python 与核心数值库版本约束，配置 JSON 随包分发。
- 将 Ray、DGL、FastDTW、Gensim、GeoSAN、ODE、统计等特定功能拆为可选安装组，并提供开发工具组；基础安装入口改为安装当前项目。
- 完整保留原依赖文件作为参考，补充可选模块到 extra 的映射、上游基线与安装说明，并忽略本地构建产物。

**作用：** 基础环境与旧功能栈有明确的安装边界；可选模型代码继续保留，但其旧依赖没有因此自动获得现代版本兼容性。

### commit3：修复现代 NumPy、CPU、配置读取和公共加载问题

**原始标题：** [fix: support modern NumPy and safe CPU checkpoint loading][commit3]。

**主要文件：** [配置解析](https://github.com/networkWaveGen/Bigscity-LibCity/blob/49acbd29179ee7c7cdde8cf8a6517325b7a17668/libcity/config/config_parser.py)、[公共检查点读取](https://github.com/networkWaveGen/Bigscity-LibCity/blob/49acbd29179ee7c7cdde8cf8a6517325b7a17668/libcity/utils/checkpoint.py)、[交通数据集](https://github.com/networkWaveGen/Bigscity-LibCity/blob/49acbd29179ee7c7cdde8cf8a6517325b7a17668/libcity/data/dataset/traffic_state_datatset.py)、[交通评估器](https://github.com/networkWaveGen/Bigscity-LibCity/blob/49acbd29179ee7c7cdde8cf8a6517325b7a17668/libcity/evaluator/traffic_state_evaluator.py)。

- 替换数据集、Node2Vec、GWNET、HIEST 等位置的 `np.float`、`np.int` 旧别名，按原别名语义使用 `float`、`int`。
- 只有请求 GPU 且 CUDA 可用时才设置 CUDA 设备；包内默认配置改用资源接口读取，支持显式 JSON 文件名及 UTF-8 配置读取。
- 引入带设备映射和 `weights_only=True` 的公共检查点读取，并接入交通、轨迹公共执行器；修正 R2、EVAR 等指标返回标量时错误调用 `.item()` 的问题。

**作用：** 解决所测核心路径在现代库和 CPU 环境中的明确启动、加载和评估问题；更完整的检查点格式与优化器恢复在 commit12 补齐。

### commit4：统一组件工厂，保留真实初始化错误

**原始标题：** [refactor: centralize component lookup and preserve initialization errors][commit4]。

**主要文件：** [组件查找函数](https://github.com/networkWaveGen/Bigscity-LibCity/blob/f36a7c5a9a5e42f93410f9ea5ca72cd7b8e655e0/libcity/_optional.py)、[模型/执行器/评估器工厂](https://github.com/networkWaveGen/Bigscity-LibCity/blob/f36a7c5a9a5e42f93410f9ea5ca72cd7b8e655e0/libcity/utils/utils.py)、[数据集工厂](https://github.com/networkWaveGen/Bigscity-LibCity/blob/f36a7c5a9a5e42f93410f9ea5ca72cd7b8e655e0/libcity/data/utils.py)。

- 新增共享 `resolve_component`，集中维护任务对应的模型包查找顺序。
- 数据集、模型、执行器、评估器创建入口统一使用组件查找，减少重复的导入和多层兜底逻辑。
- 先定位类，再在查找逻辑之外实例化；构造函数本身的异常直接向上传递。

**作用：** 模型初始化失败时能看到真实错误原因，避免把内部异常误报为组件不存在。

### commit5：减少批处理复制和转换，支持 Windows spawn

**原始标题：** [perf: avoid redundant dense batch copies and support spawn workers][commit5]。

**主要文件：** [Batch 与 padding](https://github.com/networkWaveGen/Bigscity-LibCity/blob/76053df40d06edd96714f90096c55da8bc5e1bff/libcity/data/batch.py)、[DataLoader 与 BatchCollator](https://github.com/networkWaveGen/Bigscity-LibCity/blob/76053df40d06edd96714f90096c55da8bc5e1bff/libcity/data/utils.py)。

- 数值字段通过 NumPy 汇整后使用 `torch.as_tensor`，已有 Tensor 直接调整设备和类型，减少重复转换。
- 规则数值字段减少逐样本深拷贝；变长或不透明字段保留必要隔离，padding 使用自有列表。
- 将嵌套的 collate 函数改为顶层 `BatchCollator`，便于 Windows spawn worker 序列化。

**作用：** 改善公共批处理的复制和转换开销；后续集中检查发现的样本所有权及尾批边界问题由 commit7 修复。

### commit6：减少评估阶段额外的全量预测副本

**原始标题：** [perf: remove full prediction copies during traffic evaluation][commit6]。

**主要文件：** [交通预测评估流程](https://github.com/networkWaveGen/Bigscity-LibCity/blob/7672e36895016c945a6ed8870b00c0f4bf51e174/libcity/executor/traffic_state_executor.py)。

- 预测批次转为 CPU Tensor 后收集，再以 `torch.cat` 拼接。
- 去掉整套预测先转成 NumPy、拼接后再构造 Tensor 的额外全量复制。
- 增加 `save_predictions` 配置，默认继续保存预测归档，设置为 `false` 时跳过归档文件生成。

**作用：** 减少重复转换，并保留原全数据指标和预测归档语义；仍然保留并拼接全部预测，没有改成流式评估。

### commit7：增加现代核心集中检查，修复异形样本和所有权边界

**原始标题：** [test: validate modern core and fix heterogeneous sample padding][commit7]。

**主要文件：** [19 项核心检查](https://github.com/networkWaveGen/Bigscity-LibCity/blob/cf1af9945f5d450e8ab987cfd2c47db2e1f4f226/tests/test_modern_core.py)、[批处理边界修复](https://github.com/networkWaveGen/Bigscity-LibCity/blob/cf1af9945f5d450e8ab987cfd2c47db2e1f4f226/libcity/data/utils.py)、[构建工具要求](https://github.com/networkWaveGen/Bigscity-LibCity/blob/cf1af9945f5d450e8ab987cfd2c47db2e1f4f226/pyproject.toml)。

- 新增 19 项参数化检查，覆盖可选依赖隔离、批处理、spawn worker、CPU 与配置、GRU/STGCN 前向反向、评估和 GRU 合成数据短训练。
- 修复 NumPy 2 下输入和目标形状不同时的尾样本补齐，避免用异形数组整体重复；保证稠密字段在转 Tensor 前也不会回写原样本。
- 补齐没有 padding 字段时的 BatchPAD 行为和样本字段数量检查；将 setuptools 下限调整为支持当前包配置的版本。

**作用：** 将主体修改集中验证，并修复该批检查发现的实际问题；历史真实数据精度与完整优化器恢复不属于这 19 项的通过结论。

### commit8：记录环境、静态兼容清单和首轮验证结果

**原始标题：** [docs: record model compatibility and concentrated validation results][commit8]。

**主要文件：** [已验证版本约束](https://github.com/networkWaveGen/Bigscity-LibCity/blob/ba1374e404f666f59944f18dbb6219587f5acc59/constraints/validated-py312.txt)、[静态清单工具](https://github.com/networkWaveGen/Bigscity-LibCity/blob/ba1374e404f666f59944f18dbb6219587f5acc59/tools/compatibility_report.py)、[模型兼容清单](https://github.com/networkWaveGen/Bigscity-LibCity/blob/ba1374e404f666f59944f18dbb6219587f5acc59/docs/model_compatibility.csv)、[验证说明](https://github.com/networkWaveGen/Bigscity-LibCity/blob/ba1374e404f666f59944f18dbb6219587f5acc59/docs/validation.md)。

- 保存 Python 3.12 环境的实际包版本、首轮 19 项检查结果、构建和批处理微基准记录。
- 新增不导入可选模型栈的静态扫描工具，整理 100 项任务/模型注册、所需可选组和映射问题。
- 更新现代化说明及中英文 README，记录已验证范围、兼容性待确认项和暂缓工作。

**作用：** 提供可审查的环境与验证证据。该提交当时的模型表仅为静态清点，实际导入结果在 commit13 中补充。

### commit9：整理延迟导出声明，提高可读性

**原始标题：** [style: keep lazy export declarations readable][commit9]。

**主要文件：** [执行器导出声明](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3b57551b4895ac4f1bc7e69d2dc11a7fdfdf261c/libcity/executor/__init__.py)、[模型导出声明](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3b57551b4895ac4f1bc7e69d2dc11a7fdfdf261c/libcity/model/traffic_flow_prediction/__init__.py)、[工具导出声明](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3b57551b4895ac4f1bc7e69d2dc11a7fdfdf261c/libcity/utils/__init__.py)、[调参模块格式](https://github.com/networkWaveGen/Bigscity-LibCity/blob/3b57551b4895ac4f1bc7e69d2dc11a7fdfdf261c/libcity/pipeline/tuning.py)。

- 整理各包公开导出列表的排版，保留原公开名称和导出顺序。
- 仅在顺序一致时减少重复的公开名称声明。
- 清理新调参模块的尾部空行。

**作用：** 改善阅读和 review 体验；主要是格式整理，没有新增模型算法或训练功能。

### commit10：首次生成方案执行核对文档

**原始标题：** [docs: map modernization execution to implementation commits][commit10]。

**主要文件：** [首次执行核对文档](https://github.com/networkWaveGen/Bigscity-LibCity/blob/0931d406801b221be38312c56da95460718a075c/docs/modernization_execution_zh.md)。

- 依据原方案保留 34 个方框，逐项填写已实施内容、对应提交和验证程度。
- 将当时有证据的 28 项勾选，6 项保留为部分完成，写明待补齐内容。
- 解释创建克隆和建立分支属于 Git 操作，本身不产生代码 commit，并收窄此前对完成范围的表述。

**作用：** 让代码改造与原计划逐项对应。28 项完成、6 项部分完成是该提交时的历史状态，后续由 commit11–commit15 补齐并更新。

### commit11：归档上游基线，审计依赖和已有历史实验

**原始标题：** [audit: archive upstream registrations and verify dependency removal][commit11]。

**主要文件：** [基线与依赖审计工具](https://github.com/networkWaveGen/Bigscity-LibCity/blob/78453fc10454714eeb1dc3b55dfac8c4092494ac/tools/baseline_audit.py)、[旧公开组件清单](https://github.com/networkWaveGen/Bigscity-LibCity/blob/78453fc10454714eeb1dc3b55dfac8c4092494ac/docs/audit/baseline_components.csv)、[依赖审计结果](https://github.com/networkWaveGen/Bigscity-LibCity/blob/78453fc10454714eeb1dc3b55dfac8c4092494ac/docs/audit/dependency_audit.json)、[历史实验检索记录](https://github.com/networkWaveGen/Bigscity-LibCity/blob/78453fc10454714eeb1dc3b55dfac8c4092494ac/docs/audit/historical_experiments.json)。

- 从固定上游提交导出 19 个包、220 个公开组件和原始任务配置，保存前后公开名称变化。
- 检查旧 26 项依赖的项目源码引用及当前平台现代核心的活动传递依赖，逐项记录基础、可选、开发或可移除用途。
- 检索约定的 LibCity 目录，没有发现可归档的已有历史实验产物，保存检索范围与结论；同时在开发依赖中显式加入审计工具使用的 packaging。

**作用：** 补齐基线、间接依赖和已有记录核查的证据；没有把新生成的合成测试当成历史精度结果。

### commit12：统一检查点格式和设备映射，验证真实优化器恢复

**原始标题：** [fix: restore optimizer checkpoints consistently across execution paths][commit12]。

**主要文件：** [统一检查点与恢复函数](https://github.com/networkWaveGen/Bigscity-LibCity/blob/80a990d95d4b4592d910c3de6e5186134a77a204/libcity/utils/checkpoint.py)、[旧检查点迁移工具](https://github.com/networkWaveGen/Bigscity-LibCity/blob/80a990d95d4b4592d910c3de6e5186134a77a204/tools/migrate_checkpoint.py)、[10 项检查点检查](https://github.com/networkWaveGen/Bigscity-LibCity/blob/80a990d95d4b4592d910c3de6e5186134a77a204/tests/test_checkpoint_resume.py)、[格式和迁移说明](https://github.com/networkWaveGen/Bigscity-LibCity/blob/80a990d95d4b4592d910c3de6e5186134a77a204/docs/checkpoints.md)。

- 将库内检查点读取集中到公共函数，覆盖交通、轨迹、epoch、PDFormer 初始化、GeoSAN 和调参结果读取；统一设备映射。
- 识别状态字典、模型/优化器二元组及 epoch 字典；仅模型权重不能静默用于恢复优化器。提供旧对象格式的显式迁移入口，默认保持 weights-only 读取。
- 新增 10 项检查；先实际更新产生非空 Adam 状态，再通过真实执行器保存和恢复，确认 CPU 上下一步损失、参数和优化器动量与未中断训练一致。

**作用：** 补齐模型与优化器恢复和旧格式处理。CUDA 设备参数转发已检查，真实 GPU 训练仍未执行。

### commit13：逐项验收各任务入口，修复模板映射

**原始标题：** [test: validate task entrypoints and repair template registration][commit13]。

**主要文件：** [入口审计工具](https://github.com/networkWaveGen/Bigscity-LibCity/blob/6be3388299f009e3c32b782b81697a71614b38f9/tools/entrypoint_audit.py)、[任务注册映射](https://github.com/networkWaveGen/Bigscity-LibCity/blob/6be3388299f009e3c32b782b81697a71614b38f9/libcity/config/task_config.json)、[10 项跨任务检查](https://github.com/networkWaveGen/Bigscity-LibCity/blob/6be3388299f009e3c32b782b81697a71614b38f9/tests/test_component_entrypoints.py)、[更新后的兼容状态](https://github.com/networkWaveGen/Bigscity-LibCity/blob/6be3388299f009e3c32b782b81697a71614b38f9/docs/model_compatibility.csv)。

- 逐项尝试 220 个旧公开名称及历史直接路径、100 项任务配置和组件解析、根脚本导入，并检查基础训练与可视化 CLI 的帮助入口。
- 补齐 TemplateTLP 缺失的任务映射；保留其开发模板身份，未把空模板实现成新算法。
- 创建五类任务的代表模型、执行器、评估器；检查轨迹和 ETA 数据集/编码器，并增加轨迹 GRU、DeepTTE、LINE 的损失与反向检查。
- 把兼容表更新为 82 项导入通过、16 项可选依赖缺失、2 项开发模板；补充 100 项旧模型明细和任务映射前后变化文件。

**作用：** 补齐公共结构与入口验收；可选缺项均有明确归类，模型精度及全部旧模型训练仍按实际覆盖程度记录。

### commit14：归档第二轮集中验收和构建证据

**原始标题：** [docs: record completed follow-up audits and concentrated validation][commit14]。

**主要文件：** [逐项测试和环境记录](https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/audit/validation_results.json)、[最新验证说明](https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/validation.md)、[安装及验收导航](https://github.com/networkWaveGen/Bigscity-LibCity/blob/ad6afd8f9b3e382519e5d54ed69d500d94e26eca/docs/modernization.md)。

- 归档完整 39 项检查通过、18.82 秒的结果，保存逐个测试名称、所测代码提交、实际环境和代表模型验证程度。
- 记录已安装 41 个包的元数据兼容检查、wheel 重新构建、174 个 JSON 配置及从 wheel 直接导入的结果。
- 更新说明，列出真实数据精度、真实 CUDA/Linux 和旧可选栈等尚未验证的范围。

**作用：** 形成可追溯的第二轮验收证据；此提交只更新记录和文档，没有额外改变模型或训练逻辑。

### commit15：更新全部方框，建立按时间排序的提交编号

**原始标题：** [docs: update execution checklist and chronological commit map][commit15]。

**主要文件：** [执行清单和编号映射](https://github.com/networkWaveGen/Bigscity-LibCity/blob/58f5d6785c8941b9154dde59dcc75c6c37f8d205/docs/modernization_execution_zh.md)。

- 将原来的 6 项部分完成改为已补齐，并逐项写入 commit11–commit14 的实现与验证证据；保留原方案 34 项原始内容。
- 在最前面增加短提交号、原始标题、commit 顺序编号的映射，明确 cf1af99 对应 commit7，正文改用 commit 编号。
- 增加 6 项缺口的闭环表，更新检查点、入口、兼容清单和集中验证说明，保留所有未验证范围。

**作用：** 让执行方案直接回答“哪项做了什么、由哪个提交完成、验证到了什么程度”；没有修改源代码。

### commit16：追加每个提交的修改明细（本次文档更新）

**原始标题：** [docs: append detailed change descriptions for each commit][commit16]。

**主要文件：** [执行文档的仓库副本](https://github.com/networkWaveGen/Bigscity-LibCity/blob/refactor/libcity-modernization/docs/modernization_execution_zh.md)，同步更新本地 `D:\LibCity 2.0\LibCity现代化改造实施方案的执行.md`。

- 在文档最下面增加本节，逐个说明 commit1–commit15 的主要文件、具体改动和作用。
- 补齐 commit15 的实际提交号，保留全部既有编号，并将此次文档追加记为 commit16。
- 本次仅补充说明，未改动模型、数据处理、训练逻辑和原方案的勾选状态。

**作用：** 阅读文档末尾即可按提交 review，无需只凭英文标题判断修改内容。
