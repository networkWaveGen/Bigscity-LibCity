# Current source entrypoints

| Entrypoint | Purpose |
| --- | --- |
| `run_model.py train/evaluate --config ...` | Unified training, resume and independent evaluation |
| `run_hyper.py --config ... --space ...` | Validation-only finite search, one final test evaluation |
| `libcity.pipeline.experiment.run_experiment` | Python API for the same behavior |
| `libcity.pipeline.runtime_tuning.run_search` | Python search API |
| `tools/make_runtime_fixture.py --directory ... --task ...` | Reproducible atomic data and published layered examples |

ConfigParser, custom Batch/BatchPAD, old registry factories, Executor subclasses,
standalone model/unit-test scripts, old Ray/Hyperopt CLIs and checkpoint migration
tools have been removed. Retained model/dataset source and reference JSON are
migration material; pending models are not runnable through a second pipeline.

See [examples](../../examples/README.md), [migration coverage](runtime/model-migration.csv)
and the [phase-five acceptance record](runtime/phase5_acceptance_zh.md).
