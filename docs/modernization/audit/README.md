# LibCity 现代化改造审计

本目录保存基线、依赖和入口审计脚本及其报告。

- `baseline_components.csv`：从固定上游提交导出的 19 个包、220 个公开组件名称及其源码目标。
- `baseline_task_config.json`：原始任务配置快照，包含 100 项任务/模型注册；包含上游已有的模板缺项，未伪装成已修复版本。
- `baseline_models.csv`：100 项旧模型名称、源码目标及数据/执行器/评估器/编码器映射；`task_registration_changes.json` 单独列出本轮任务映射变化及是否删除允许项。
- `registration_changes.json`：比较基线和当前的包级导出。显式导入改造移除了这些兼容导出，配置中仍使用原有注册名称。
- `dependency_audit.json`：旧 26 项依赖逐项分类，同时检查全部项目 Python 导入及当前平台现代核心的已安装 `Requires-Dist` 传递依赖闭包。
- `historical_experiments.json`：检索范围、检查文件数、候选历史产物及是否有历史精度基线。
- `baseline_audit_summary.json`：上述审计的汇总。
- 新版入口审计输出 `implementation_imports.csv`，并更新 `registration_imports.csv`、`script_imports.csv` 和 `entrypoint_audit_summary.json`。保留的 `public_imports.csv` 是改造前的包级导入记录。
- `model_compatibility.csv` 汇总模型依赖、配置和导入状态。
- `validation_environment.json` 和 `validation_results.json` 保存验收环境与测试记录。

当前 CSV、JSON 仍是显式导入改造前的归档。安装项目 `dev` 可选组后，可以从仓库根目录运行以下命令。报告默认写入本目录。

```shell
python docs/modernization/audit/baseline_audit.py
python docs/modernization/audit/entrypoint_audit.py
python docs/modernization/audit/compatibility_report.py --runtime-report docs/modernization/audit/registration_imports.csv
```

基线审计直接从固定提交 `5a6391d` 读取旧组件配置和依赖清单。

依赖审计确认：TensorBoardX、aioredis、aiohttp、idna 在本轮项目源码中没有直接用途，也不在当前平台现代核心依赖闭包内，因此不保留为基础安装的直接依赖。Ray 等可选栈若需要其中某项，应由该可选栈自己的依赖声明安装，不能据此强制所有基础实验安装它们。flake8、pep8-naming 属于历史静态检查工具，不是运行核心；pytest 属于 `dev`。其他旧依赖均逐项对应现代核心或已保留的可选组。现代基础安装没有未分类的旧直接依赖。

已有实验检索覆盖原 LibCity 参考副本、`LibCity2.0-main/LibCity` 和当前工作副本的缓存、日志目录，排除约定不处理的 AgentCity。检索时没有找到可归档的历史实验日志、检查点或指标产物。`.runtime-tests` 中的结果是本轮生成的合成测试，不能作为历史真实数据精度基线。该项完成的是“保存已有可用记录”的核查与结论归档，不表示获得了原本不存在的历史精度结果。

此审计读取 Git 提交中的源码、普通文件内容及包元数据，不做文件哈希校验。依赖闭包结论针对已记录的 Windows CPU 环境，不构成所有平台和可选栈的完整锁定文件。
