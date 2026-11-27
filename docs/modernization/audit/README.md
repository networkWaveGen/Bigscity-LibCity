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

这些 CSV/JSON 是早期审计归档，不代表阶段五后的可执行入口。使用旧 ConfigParser/registry 的审计脚本已随旧 API 删除；如需重现历史审计，请在独立 checkout 的 3dcae47 或更早版本执行对应版本脚本。

当前迁移清单用 `python tools/build_runtime_migration_table.py` 生成，完整回归用 `python -m pytest -q` 执行。当前验收记录见 [阶段五](../runtime/phase5_acceptance_zh.md)。
