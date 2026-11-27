# Source examples

Run with Python 3.11+ from a source checkout after `python -m pip install -e .`.
The examples need no downloaded datasets or pytest. The generator copies the
published JSON config and creates small atomic data in the directory you choose.
Use a fresh output directory when repeating the commands.

```powershell
python tools/make_runtime_fixture.py --directory ../libcity-demo
python run_model.py train --config ../libcity-demo/gru.json --set runtime.exp_id=train
python run_model.py evaluate --config ../libcity-demo/gru.json --checkpoint ../libcity-demo/runs/train/checkpoints/best.pt --set runtime.exp_id=evaluate
python run_model.py train --config ../libcity-demo/gru.json --resume ../libcity-demo/runs/train/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=resumed
python run_hyper.py --config ../libcity-demo/gru.json --space hyper_example.json --max-trials 4
```

Five task examples:

| `--task` | Generated JSON | Result |
| --- | --- | --- |
| traffic_state_pred | gru.json | Forecast metrics and optional predictions.npz |
| traj_loc_pred | rnn.json | Full-vocabulary Recall/MRR and predictions.json |
| eta | deeptte.json | Metrics in seconds and predictions.json |
| road_representation | line.json | Held-out edge metrics, embeddings.npy, node_ids.json |
| map_matching | hmmm.json | Direct route metrics and optional predictions.json |

For example, `python tools/make_runtime_fixture.py --directory ../eta-demo --task eta`
creates data for `python run_model.py train --config ../eta-demo/deeptte.json`.
HMMM uses `evaluate --config .../hmmm.json` without a checkpoint and has no training
or resume mode. The other tasks use the same train/evaluate/resume lifecycle above.

The Python equivalent is `from libcity.pipeline.experiment import run_experiment`.
For the CRANN spatial/temporal example, generate with
`python tools/make_runtime_fixture.py --directory ../crann-demo --model CRANN`,
then use `../crann-demo/crann.json` with the same train/evaluate/resume commands.
CRANN explicitly declares long temporal history and spatial grid dimensions;
only `known_future_columns` become future covariates. Other external observations
are restricted to historical inputs.

Road attribute autoencoders have their own named reconstruction protocol. Generate
`python tools/make_runtime_fixture.py --directory ../road-reconstruction --model ChebConv`
or use `--model GAT`, then train/evaluate/resume with the generated JSON. Inputs
contain the full observed graph and node attributes; disjoint node masks select
the reconstruction targets. Scaling is fitted on training nodes. The run records
`road_reconstruction`, raw attribute MAE/RMSE, and full-node embeddings; these
metrics describe transductive reconstruction and do not measure downstream road
classification. Graph training uses a single full-graph batch. Feature columns
and categorical columns are configurable through the data block.

For all three temporal branches of ASTGCN/MSTGCN, generate with `--model ASTGCN`
or `--model MSTGCN`. The published hourly fixture covers weekly, daily and recent
segments. Runs record the CPT sampling layout and `history_span_points`; the
number of input samples differs from their span on the original timeline.

`--model MegaCRN` creates a memory-augmented forecast example with explicit known
future time features. Its auxiliary memory losses and curriculum targets are
used only by the training step; ordinary forward/evaluation returns predictions
without labels. Resume restores the curriculum counter and random generators.

`--model TESTAM` creates a three-expert forecast example. Its ordinary forward
returns a prediction tensor; expert/routing supervision is training-only.
Time-of-day comes from metadata, and the example uses a two-step forecast from
four historical steps. The original warmup/periodic learning-rate curve is
configured explicitly with `inverse_sqrt_cosine`; keep `scheduler.model_dim`
aligned with the intended optimizer scale when changing the model width.

`--model ToGCN` and `--model SimST` generate graph/temporal examples with the same
train/evaluate/resume commands. ToGCN trains with explicit teacher targets while
inference reads only history. SimST creates incoming/outgoing neighborhood features
from the configured graph and keeps the same full-node meaning for every batch
size. Its input window and forecast window may differ.

`--model STNN` uses differentiable factor interpolation (`lookup=linear`) and
explicit time/node axes. This intentionally changes the old hard lookup, whose
input projection received no gradients. `lookup=hard` is available for comparison;
both modes use the actual node count and history length.

`--model TGCLSTM` generates the required FFR edge columns and metadata. By default,
the model reads `data.graph.reachability_column=FFR_20min`; set it to another
existing column or JSON `null` for plain graph filters. Missing edges have zero
reachability. Inference uses history and explicit known future features only.

`--model STTN` creates a spatial/temporal transformer example. Its graph is an
immutable buffer, so repeated inference starts from the same graph. Positional
capacity is inferred from metadata unless `position_size` is explicitly supplied.

`--model TimeMixer` creates a multi-scale example with explicit normalization.
`channel_independence` and `use_norm` are JSON booleans. Average, max and learned
convolution downsampling use their actual scale lengths; `dft_decomp` selects
`top_k` temporal frequencies independently per sample/channel. Normalization
statistics stay local to the forward call, and forecasts contain only target
channels. Unused legacy time-embedding settings are no longer accepted.

`--model STG2Seq` keeps the six-layer long-term and three-layer short-term graph
stacks, with explicit teacher targets during training. Inference is autoregressive
and takes only declared future covariates for its attention queries. Graph and
attention operations preserve batch/time/node axes; these correct intentional
differences from the old reshape-based implementation.

The complete supported-model inventory and migration details are in the
[runtime guide](../docs/modernization/runtime/README_zh.md). Small examples prove
execution and reproducibility, not real-city benchmark accuracy.
