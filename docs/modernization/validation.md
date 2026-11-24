# Modern core validation

This page archives the 2026-11-24 Windows CPU acceptance run. The commands
below use the current paths and audit checks. The archived reports have not
been regenerated since the explicit-import refactor.

## Environment and reproducible commands

Windows 11, Python 3.12.13, CPU PyTorch 2.15.0, NumPy 2.5.5, pandas 3.1.1,
SciPy 1.18.1 and scikit-learn 1.9.1. Exact versions and CPU details are in
[the audit results](audit/validation_results.json). Validated core constraints
remain in [validated-py312.txt](../../constraints/validated-py312.txt).
No real CUDA or Linux run was performed.

Install the core and development tools, then run the audit and concentrated checks:

```shell
python -m pip install -e '.[dev]' -c constraints/validated-py312.txt
python docs/modernization/audit/baseline_audit.py
python docs/modernization/audit/entrypoint_audit.py
python docs/modernization/audit/compatibility_report.py --runtime-report docs/modernization/audit/registration_imports.csv
python -m pytest tests -q --disable-warnings --maxfail=2 --basetemp ../.runtime-tests/modernization-acceptance --junitxml ../.runtime-tests/modernization-acceptance.xml
python -m build --wheel --no-isolation
```

The actual run used absolute paths for the same workspace-owned temporary
directories. Create the parent of `--basetemp` first and reserve that exact
directory for pytest, which clears it on rerun. The local environment was
created without pip; its package metadata check used `uv pip check --python ...`
and passed for all 41 installed packages.

## Concentrated results

The follow-up concentrated run passed **39 tests in 18.82 seconds**, with one
warning and no failures, errors or skips. It includes the original 19 checks,
10 checkpoint checks and 10 additional cross-task checks. The machine-readable
record lists each test case and the tested code commit.

Original core coverage is retained:

- Core imports without loading Ray, DGL, TorchText, FastDTW or Gensim.
- Dense values, dtype and source ownership, padding, ragged fields, malformed
  samples, heterogeneous X/y shapes and incomplete final batches.
- One Windows spawn DataLoader worker, CPU selection and packaged configuration
  loading from another working directory.
- Constructor errors propagate through the component factory.
- Traffic GRU and STGCN forward/backward and state-dict checkpoint loading.
- All ten traffic metrics with zero labels and an uneven last batch, in both
  single-horizon and cumulative-average modes; prediction archive values and
  the archive switch.
- Synthetic atomic files through the actual GRU pipeline: preprocessing, one
  training epoch, best-checkpoint reload, model saving and two-horizon metrics.

New checkpoint coverage creates nonempty Adam states through real updates,
saves/restores through traffic, trajectory and epoch executor entrypoints, then
compares the next loss, parameters and optimizer moments with uninterrupted
training. It also checks model-only loading, rejection of silent optimizer reset,
device-argument forwarding, historical state wrappers and explicit object-pickle
migration. See [checkpoints.md](../checkpoints.md). CUDA argument forwarding is
tested; actual CUDA execution is not.

New task coverage creates actual model/executor/evaluator objects for traffic
GRU, trajectory GRU, ETA DeepTTE, map matching HMMM and road LINE. Trajectory and
ETA dataset/encoder factories are exercised; trajectory GRU, DeepTTE and LINE
also have finite-loss/backward checks. HMMM is a traditional model and does not
require backpropagation. See [entrypoints.md](entrypoints.md).

The wheel was rebuilt successfully. It contains all **174** source JSON
configurations, including the repaired TemplateTLP mapping. Direct import from
that wheel in isolated Python mode passed using the existing validated
dependency environment. This is not a separate clean dependency installation.

## Archived registration and dependency audit results

- The fixed upstream snapshot is archived as 220 public components in 19
  packages and 100 task/model entries, with explicit before/after mapping files.
- Every public export and its historical direct path was attempted: **200
  import/identity checks passed, 20 require absent optional dependencies**.
- Every task registration loaded its actual configuration and attempted all
  component resolutions: **84 passed, 16 require absent optional dependencies**.
- Root script imports: **19 passed, 3 require tuning dependencies**. Core
  training and visualization CLI `--help` checks passed.
- There are no unclassified core import failures or syntax errors. No old
  public export or allowed model was removed. TemplateTLP's missing mapping was
  repaired; TemplateTLP and TemplateTSP remain developer templates.
- All 26 historical requirements were classified against Python source imports
  and the active installed transitive core dependency graph. No unexplained
  dependency removal or missing core distribution remains.
- No existing historical experiment artifacts were found in the specified
  LibCity directories. The search scope and absence are recorded, rather than
  fabricating historical accuracy results.

The compatibility CSV now contains **82 import-passed entries, 16 optional
dependency gaps and 2 template-only entries**. All 100 configurations were
checked, including the optional entries. Per-model training coverage remains
separately recorded above and in the machine-readable validation result.

## Collation microbenchmark

The earlier comparison remains unchanged; no new training-speed claim is made.
It compares upstream Batch/deep-copy collation with the updated collator on
64 float32 samples, `X=(12,128,2)` and `y=(3,128,1)`, one CPU Torch thread,
three warmups and the median of 20 runs. Outputs were compared elementwise.

| Measurement | Upstream | Updated |
| --- | ---: | ---: |
| Collation and tensor conversion median | 0.227 ms | 0.057 ms |
| Peak allocations tracked by tracemalloc | 1,693,296 bytes | 888,064 bytes |

This is not end-to-end training performance. Tracemalloc excludes native Torch
allocations and GPU memory. Evaluation still concatenates complete predictions
to preserve global metric semantics, while avoiding extra NumPy/Torch copies.

## Remaining limits and deferred work

Real benchmark accuracy parity against historical PyTorch, actual CUDA/Linux
execution, runtime certification of all optional models and end-to-end training
performance remain unverified. Missing old classes in arbitrary object pickles
cannot be reconstructed automatically. Original checkpoints do not contain
scheduler/RNG state, so arbitrary stochastic training is not promised to resume
bitwise identically.

TorchText/DGL/DTW/embedding stack replacement, legacy Ray API migration,
configuration/path redesign, stable experiment result APIs, broader executor
deduplication, streaming metrics, window sampling rewrites, compiler/distributed
training and AgentCity remain deferred as agreed.
