# LibCity 提交审查待办

审查范围从 `8b68efbec0d4e17b78f8f9eb2a565d7d56f8feaa` 开始。下文的 `commit0` 等编号对应重排前的 Git 历史。

`docs/modernization/` 用于保存现代化改造分支的交流文档。后续把代码改动合入主分支时，排除这些交流文档及只指向这些文档的分支专用链接。

## commit0 · `8b68efb` · 按需加载及调参隔离

允许调整旧的包级 Python 导入路径。保留配置中的模型和组件名称，并继续让普通训练独立于 Ray 等可选依赖。

- [x] 将仓库内依赖包级导出的调用方迁到具体实现模块。例如使用 `from libcity.model.traffic_speed_prediction.STGCN import STGCN`。留空只用于转发的 `__init__.py`，删除重复的 docstring。
- [x] 让组件工厂根据 `libcity/registry.py` 中的注册项直接加载实现。配置继续使用 `"model": "STGCN"` 等现有名称。已移除包级 `__getattr__` 和转发映射。
- [x] 删除 `libcity/pipeline/pipeline.py` 末尾转发 `hyper_parameter` 和 `parse_search_space` 的代码，收掉多余空行。`pipeline.py` 以 `objective_function()` 结束，两个调参函数从 `tuning.py` 导入。
- [x] 同步三个根目录脚本的导入。
  - `run_model.py` 使用 `from libcity.pipeline.pipeline import run_model`。
  - `run_hyper.py` 使用 `from libcity.pipeline.pipeline import objective_function`。
  - `hyper_tune.py` 使用 `from libcity.pipeline.tuning import hyper_parameter`。
- [x] 删除 `_optional.py`。可选依赖组由 `pyproject.toml` 声明，执行器只在进入调参分支时导入 Ray。缺包与 API 不兼容的错误由 Python 原样报出。
- [x] 同步测试、审计脚本和文档。39 项 pytest 检查已通过，普通训练入口可在没有 Ray 的环境里导入。审计脚本按要求未运行，CSV 和 JSON 保留上一轮结果。

## commit1 · `e214b0c` · 安装与依赖配置

- [x] 把改造说明、执行记录、入口验收及集中验证归入 `docs/modernization/`，同步更新分支文档中的引用。
- [x] 保留审计，将三个脚本和审计输入、报告归入 `docs/modernization/audit/`。基线审计从 `5a6391d` 读取旧依赖，入口和兼容性审计使用显式路径及当前注册表。
- [x] 保留 `pyproject.toml` 的安装方式和 wheel 构建能力。本轮验证了正常解包安装以及直接从未解包 wheel 读取自带配置。

## commit2 · `49acbd2` · 包内默认配置路径

- [x] 用 `importlib.resources.files(__package__)` 获取一次当前配置包的资源根目录，从同一个根目录读取各类默认 JSON。源码、正常安装和直接导入 wheel 的包内配置读取都与当前工作目录无关。

## commit3 · `f36a7c5` · 查找与实例化分开

- [x] `load_component()` 只加载并返回类，docstring 为 `Return the registered class for the given name.`。工厂在下一步实例化，让类的 `__init__` 报错直接传给调用方。
