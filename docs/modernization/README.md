# LibCity 当前现代化运行时

- [源码用法与五类示例](runtime/README_zh.md)
- [阶段五验收](runtime/phase5_acceptance_zh.md) 与 [执行/Commit 映射](runtime/phase5_execution_zh.md)
- [100 条模型迁移清单](runtime/model-migration.csv)：54 个入口对应 56 条已迁移记录，44 条 pending；最新执行范围见 [主线剩余问题记录](runtime/remaining-work_execution_zh.md)
- [旧测试迁移映射](runtime/phase5-test-migration.json) 与 [移除/参考配置清单](runtime/phase5-retirement.json)

仅新入口提供实验执行。旧 ConfigParser、Batch、Executor、registry 工厂和检查点迁移工具已删除。reference 目录中的旧配置、经典基线数学代码以及保留的待迁移模型源码不是第二套受支持入口。

以下内容为早期现代化基线的历史记录，描述当时保留的组件与检查，不适用于阶段五已经退役的接口。当前安装与运行以以上链接及根目录 README 为准。

---

# LibCity modernization

Development branch: `refactor/libcity-modernization`.
Upstream baseline: `LibCity/Bigscity-LibCity`, master commit
`5a6391d41944e937f2c15e9be85ab7f40ac8b23e` (2024-12-18).
Existing model names continue to work in experiment configuration. Python
callers should import components from their implementation modules.
AgentCity is outside this change.

Phases 3–4 add target-time splitting, five representative task pipelines and
validation-only finite search. See the [implementation record](runtime/phases34_execution_zh.md)
and [acceptance report](runtime/phases34_acceptance_zh.md). The full
[100-entry migration inventory](runtime/model-migration.csv) distinguishes supported
representatives from pending models; it does not certify the whole model catalog.

The source CLI now uses `run_model.py train/evaluate --config ...` and layered
JSON configuration. See the [new runtime guide](runtime/README_zh.md) and
[phase acceptance record](runtime/execution_zh.md). Historical executor audits
below describe the retained migration baseline, not the new CLI contract.

The follow-up [runtime fixes](runtime-fixes_zh.md) address evaluation sample
accounting, checkpoint/mode controls and CPU placement. That work has its own
70-case validation record; the earlier audit reports below remain archived.

The subsequent [follow-up fixes](followup-fixes_zh.md) cover CSTN tail batches,
validation overrides, JSON mode flags, trajectory best-state restoration and
repeated dataset access. The amended change also isolates data cache configurations,
preserves repeated-user ETA exports and handles zero normalization scales.
The latest record contains 129 passing cases, including 27 additional regressions;
the prior 102-case and 70-case records are unchanged.

## Core installation

Use Python 3.12 for the primary development environment; Python 3.11 is also
accepted by the package. Create a separate environment and install from this
checkout:

```shell
python -m pip install -e .
```

Install a PyTorch build suitable for the machine's CPU or CUDA runtime. The
package specifies compatible version ranges; validation records contain the
exact combination actually exercised. The audit reads the historical requirements
from the fixed upstream commit.

Default JSON configurations are package resources and can be read from any
working directory. Dataset files still follow the configured experiment layout.

## Python imports

```python
from libcity.model.traffic_speed_prediction.STGCN import STGCN
from libcity.pipeline.pipeline import run_model
from libcity.utils.dataset import preprocess_data
```

The component registry loads the implementation selected by a configuration
name. Ray Tune is loaded through `libcity.pipeline.tuning`.

## Optional components

Only the requested model, dataset, encoder, executor and evaluator are loaded.
For example, basic GRU training does not require Ray, DGL, FastDTW or TorchText.
Install a feature from this checkout with `pip install -e '.[graph]'`, replacing
`graph` with the appropriate extra below.

| Extra | Components | Modern-environment status |
| --- | --- | --- |
| tune | Ray Tune, Hyperopt, Bayesian search | Legacy Ray APIs remain; compatibility pending |
| graph | DGL models such as GeomGCN and STMGAT | Compatibility pending |
| dtw | FastDTW dataset preprocessing | Compatibility pending |
| embeddings | Gensim word and graph embeddings | Node2Vec/DeepWalk use Gensim>=4.4 with NumPy 2; other pending consumers still need migration |
| No extra for GeoSAN | Native quadkey vocabulary and geographic sampling | Uses the core installation; TorchText/NLTK and the obsolete geosan extra were removed |
| ode | Neural differential equation models | Compatibility pending |
| transformer | Models using einops | Compatibility pending |
| multispans | MultiSPANS graph and community detection dependencies | Compatibility pending |
| wavelet | Wavelet preprocessing | Compatibility pending |
| pdformer | PDFormer preprocessing dependencies | Compatibility pending |
| statistics | Historical statistical baselines | Compatibility pending |

Making a dependency optional does not make its old API compatible with current
PyTorch or NumPy. These components are preserved rather than rewritten or
silently replaced. Select them only in a suitable environment. No `all` extra
is provided because legacy optional stacks can require incompatible versions.

See [the compatibility inventory](audit/model_compatibility.csv) for archived config and
import outcomes, [the entrypoint audit](entrypoints.md) for all task registrations,
and [the validation record](validation.md) for 39 concentrated checks. Import
success does not establish that every model trains correctly. Historical source,
dependency and existing-result audits are in the [audit directory](audit/README.md).
Checkpoint formats, optimizer resume and explicit migration are described in
[checkpoints.md](../checkpoints.md).

`TrafficStateExecutor.evaluate()` saves prediction archives by default. Set
`save_predictions` to `false` in a user JSON config to skip writing those files.
Executors with their own `evaluate()` method, such as STSSL and PDFormer, keep
their own archive behavior. Extending the switch to those executors is deferred.

## Review and validation

Changes are committed by topic, without co-author trailers. Syntax and import
checks run after a coherent batch; forward/backward and training checks are
collected after the main changes. Model compatibility status distinguishes
static registration, import, construction and actual execution.

Test shared behavior once at its common entry point. Where a model, dataset, or
executor overrides that behavior, exercise a representative override with the
same assertions. The traffic split regression covers both the common dataset
and CRANN's separate split for temporal, spatial, and external inputs.

Deferred work includes TorchText replacement, tuning-backend replacement,
AgentCity integration, a new result API, broad path/config redesign, memory
mapping, rewritten window sampling and compiler/distributed-training changes.

The distribution format is still open. The main README installation and import
guidance, clean Linux CPU installation checks, and a short run on a real traffic
dataset are deferred.
