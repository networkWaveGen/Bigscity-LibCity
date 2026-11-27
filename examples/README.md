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

`--model DeepMove` creates a trajectory example with masked historical attention.
History contains only the same user's sessions completed before the current trip
started. Empty history becomes zero context without dropping benchmark samples.
RNN/GRU/LSTM and dot/general/concat attention use the standard ranking lifecycle.

`--model STRNN` creates an example with distinct POI geometry. Every candidate has
its own distance-dependent state. The default query is the last observed time;
actual next-event timestamps and target coordinates are excluded. Automatic time
bounds use training prefixes, while spatial bounds use the declared static POIs.
The model returns full-vocabulary log probabilities for common NLL/ranking.

`--model HSTLSTM` creates an observed-interval trajectory example. Time gaps are
seconds and distances are WGS84 geodesic kilometres. Slot bounds use training
prefixes only; validation/test values are clamped without refitting. The corrected
ST-LSTM output uses its updated memory cell. Padded steps leave the state intact.

`--model ATSTLSTM` creates a personalized attentive trajectory example. It uses
training-only popularity for negative sampling, excludes the positive location,
and preserves the normalized pairwise ranking objective. `negative_samples` is
under `training.loss`, with `name=normalized_bpr` selected by the named protocol.
Training scores only selected candidates; evaluation ranks the full vocabulary.
Distances use the last known position and static POIs. Actual next-event location
and time are excluded; a known query offset can be supplied explicitly in seconds.
Validation loss is deterministic full-vocabulary normalized BPR, rather than NLL.

`--model SERM` creates a semantic trajectory example with a small synthetic vector
file. In the dataset's `config.json`, set `word_vector_file` to a headerless GloVe
text file (absolute, or relative to the dataset directory). `semantic_column`
defaults to `venue_category_name` in `.geo`. Pretrained vectors stay intact and
trainable; only words used by static POI categories are retained. Duplicate words
keep their multiplicity, and categories without known words become zero vectors.
POIs use their real IDs rather than DataFrame row numbers. The same train,
evaluate and resume commands apply, with full-vocabulary NLL/ranking. Resume
rejects changes to the referenced vector file before updating parameters.

`--model STAN` creates a two-level spatial-temporal attention example. Hour-of-week
and user tokens reserve zero for padding. The query time defaults to the last
observation, with an optional explicitly known `query_offset_hours` input. Temporal
bounds use training prefixes; static spherical geometry uses a 6371 km radius.
`max_len` must cover every actual prefix and is checked before training. The named
`stan_ranking` protocol uses `training.loss.name=sampled_softmax` and shared sampled
candidates with unique positive labels. Requests for too many negatives are capped;
full-vocabulary validation/test use deterministic NLL and ranking. Inference does
not require labels, next-event times or a dense POI distance table.

`--model CARA` creates a contextual recurrent ranking example. It restores the
author implementation's gate activations and small normal embedding initialization.
`history_window=0` uses the complete observed prefix; a positive value selects the
most recent observations. Gaps use actual timestamps in hours, including midnight
transitions. The optional `query_offset_hours` is explicitly known input; actual
next-event times and locations are never inference features. The `cara_ranking`
protocol uses per-sample BPR (`training.loss.name=bpr`) and bounded negative sampling.
Validation/test rank the full vocabulary with deterministic mean BPR. Prefix states
are reused across candidate chunks rather than replaying the sequence per candidate.

The complete supported-model inventory and migration details are in the
[runtime guide](../docs/modernization/runtime/README_zh.md). Small examples prove
execution and reproducibility, not real-city benchmark accuracy.
