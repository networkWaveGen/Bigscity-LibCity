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
or use `--model GAT` / `--model GeomGCN`, then train/evaluate/resume with the generated JSON. Inputs
contain the full observed graph and node attributes; disjoint node masks select
the reconstruction targets. Scaling is fitted on training nodes. The run records
`road_reconstruction`, raw attribute MAE/RMSE, and full-node embeddings; these
metrics describe transductive reconstruction and do not measure downstream road
classification. Graph training uses a single full-graph batch. Feature columns
and categorical columns are configurable through the data block.

GeomGCN uses the core installation and native PyTorch partition aggregation,
without DGL. It preserves LibCity's four-layer autoencoder with separate road and
self edges; it does not construct the original paper's embedding-space
neighborhoods. Source and destination normalization use the full incoming degree,
including exactly one self edge per node. Merge settings accept `cat`/`mean`;
`embedding_dim` is a per-head, per-partition width, so concatenation in layer two
increases the exported embedding width. The decoder follows actual widths and
always averages the final partitions/heads to reconstruct the input attributes.
Defaults retain unscaled inputs, dropout 0.5 and the original final ReLU.

For all three temporal branches of ASTGCN/MSTGCN, generate with `--model ASTGCN`
or `--model MSTGCN`. The published hourly fixture covers weekly, daily and recent
segments. Runs record the CPT sampling layout and `history_span_points`; the
number of input samples differs from their span on the original timeline.

For grid forecasts, generate with `--model STResNet` or `--model ACFM` and use
the resulting `stresnet.json` / `acfm.json` with the common train/evaluate/resume
commands. Atomic `.geo` rows need unique IDs and zero-based `row_id`/`column_id`
covering a complete rectangle. `.grid` observations are aligned by time/cell;
`.dyna` with `entity_id` is also accepted. Neither CSV row order nor ID magnitude
defines the convolution layout. Predictions follow the original node order.

The named `grid_cpt` protocol defaults to one prediction step and three recent
frames plus the previous day's matching frame. Closeness counts raw frames,
period counts days, and trend counts weeks; `interval_period`/`interval_trend`
multiply those day/week spacings. Inputs list closeness/period/trend, nearest
first. Explicit `output_window` supports direct multistep prediction. This differs
from the hourly CPT protocol of the graph models and is recorded in run metadata.

`X` contains observed history. Global `history_features` contain historical
calendar/external values; global `future_features` contain only calendar values
and explicitly declared `known_future_columns`. ACFM accepts different historical
and future external widths. Actual future observations never become model input.
External values remain unscaled; target scaling fits the training interval only.
The raw-array cache is shared across horizon/split changes, checks `.grid` source
versions, and rebuilds missing/malformed arrays. Changing source data rejects
full-state resume. Singleton 1x1-grid tail batches use running BatchNorm statistics.
ACFM supports `split`, `split-chans`, `concat`, `cpt` and compatible
`cpt-sameoutput` residual modes; the last requires each active count times hidden
width to equal two. The unused legacy `merge_mode` option is removed. Common
variants remain deferred. Tiny runs validate execution rather than city accuracy.

`--model CONVGCN` and `--model MultiSTGCnet` generate periodic **point graph**
examples. Use `convgcn.json` / `multistgcnet.json` with the common
train/evaluate/resume commands. These models consume atomic `.geo`/`.rel`/`.dyna`;
they do not require a rectangular grid. CONVGCN's named `convgcn_windows` protocol
defaults to ten observations per window and service shifts of 36/180 raw frames.
The `seasonal_windows` input lists trend/period/recent windows, chronological
within each window. `period_steps`/`trend_steps` are explicit frame counts, not
automatically calendar day/week spacings. Target channels remain distinct and the
final complete forecast window is retained. Node-channel Conv3d uses explicit
axis conversion rather than the old mixed reshape.

MultiSTGCnet's `multi_stgc` protocol uses `frame_cpt`: recent raw frames, matching
daily and weekly frames, ordered closeness/period/trend and nearest first inside
each segment. `interval_period` and `interval_trend` multiply day/week spacing.
Distance supports come from finite listed `.rel` distances with
`data.graph.name=distance_bands`; select `distance_column` explicitly when needed.
Unlisted edges are excluded, maximum distances enter the far band, and equal
distances enter the near band. Explicit zero distances are valid. Symmetrizing
takes the minimum observed directional distance; self loops have zero distance.
An explicit binary graph instead assigns unit distance to listed edges. The
recent/daily/weekly recurrent blocks have separate parameters; absent branches
are omitted. Both models support explicit multistep and multiple target outputs.
The original dense node-channel/global-LSTM scaling costs remain; the examples
certify small lifecycles rather than large-city performance or paper accuracy.

`--model DKFN` creates a contiguous-window example with dual graph/self recurrent
streams. Use `dkfn.json` with the common train/evaluate/resume commands. Graph
powers and filters are registered buffers. Variances are computed per sample,
not over other batch members. The authors' asymmetric coefficient formula is
retained; `softplus(c)` is the positive physical coefficient initialized to one.
The stored `c` parameter is its unconstrained coordinate. Scalar/zero-variance
streams use finite fallbacks. Multistep forecasts roll their predictions into
the target channels, retain the last observed unknown covariates, and insert
only declared known future channels. Forward never reads labels. Model settings
are `hops`, `clamp_adjacency` and `threshold`; fitting defaults to RMSprop.

`--model ResLSTM` creates a longer periodic point-flow example. Run
`reslstm.json` through the common lifecycle. It needs at least two raw flow
columns; `model.params.flow_indices` selects two (default 0/1). Historical
calendar/.ext covariates are separate from raw observations, and future values
or labels never enter the model. `reslstm_windows` defaults to five frames per
window and service-frame shifts 72/360. Three periods are explicit convolution
channels. Node-axis LSTMs keep each batch sample independent. SAME pooling and
derived flattened widths support odd/even nodes and history lengths, including
singletons, multiple forecast steps and target channels. Base/recurrent widths
and layers are configurable; defaults preserve 32/128/two-layer LibCity blocks.

For ResLSTM/CONVGCN service windows, overnight closures may occur **between**
the three windows; every selected window and the recent-history-to-target span
must still be contiguous. Atomic file boundaries are not crossed. Ordinary and
calendar-period windows retain full-span continuity checks. The chosen policy
is recorded; affected gap datasets carry a semantic marker for resume checks.
The old mixed period axes, shifted pooling and batch-as-sequence behavior are
explicitly corrected. The LibCity external recurrent branch is retained rather
than claiming an exact port of the authors' different Keras external branch.

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

`--model LSTPM` creates a long/short-term session example with distinct POI geometry.
Only sessions completed before the current trip are history; explicit session
lengths preserve the carried historical LSTM state and two-level attention.
Time-slot Jaccard similarity uses atomic training observations only. Spatial
parents choose the nearest earlier observation, resolving metre-rounded ties in
favour of the latest earlier point. History distances retain LibCity's arithmetic
session-centroid definition with a 1 km floor. Empty history becomes zero context;
first-trip samples remain in the benchmark. Full-vocabulary mean NLL replaces the
old summed NLL including the padding class. Train/evaluate/resume use the shared
runtime; the unused legacy dropout option is removed.

`--model GeoSAN` creates a geographic self-attention example using the core
installation. Quadkey level/ngram (default 17/6) are data settings. A plain
vocabulary replaces TorchText/NLTK, preserving PAD=0 and frequency/lexical ordering;
it is built from static POIs rather than future check-in frequencies. The
`geosan_ranking` protocol configures `training.loss.name=weighted_binary`,
`negative_samples`, `num_nearest`, `temperature` and `exclude_visited`. Geographic
negatives use haversine radians, cap the neighbour pool, remove the positive and
sample uniformly with replacement. Visited filtering uses training observations;
exhaustion falls back to geographic nonpositives. Single-POI data use only the
positive binary term. Evaluation ranks all POIs and reports deterministic weighted
binary loss, not NLL or the old sampled novelty benchmark. Source encoding is
reused across candidate chunks; optional decoder attention and corrected user/time
fusion are explicit model settings. Common train/evaluate/resume commands apply.

`--model TTPNet` creates an ETA example with synthetic static road embeddings and
departure-known traffic. Actual data need JSON `embeddings` in `.geo`, plus
`grid_len`, three four-value profiles (`speeds`, `speeds_relevant1`,
`speeds_relevant2`) and the seven-value `speeds_long` in `.dyna`. Every route point
must have `traffic_snapshot_time` at/before departure. Dataset metadata may select
another column with `traffic_snapshot_time_column`; this timestamp represents the
latest availability of all supplied profiles, not the actual arrival time.
`speed_unit` supports `km/h` (default) and `m/s`; `grid_len_unit` and `distance_unit`
support `km` (default) and `m`. Source speed imputation and graph embedding training
remain upstream preprocessing. Static embedding values stay trainable and use real
road IDs. All normalizers fit training routes only. The `ttpnet_eta` protocol uses
per-route cumulative relative RMSE for training and whole-trip MAPE for validation;
`training.loss.eps` defaults to one second. Metrics/predictions are exported in
seconds. The default minimum of three points matches the shared ETA benchmark;
explicit `data.min_route_length=2` also supports two-point routes. Padding never
enters the bidirectional state or cumulative objective. Train/evaluate/resume use
the common runtime, including dropout RNG restoration.

`--model Node2Vec` and `--model DeepWalk` create CPU examples using the external
Word2Vec backend. Install `pip install -e '.[embeddings]'` for fitting (Gensim>=4.4,
NumPy 2 compatible). Use the normal `train` command, then `evaluate --checkpoint`
with the returned `checkpoints/embeddings.pt`; evaluation needs no Gensim import.
The `road_walk` protocol fits the whole graph without a validation/test split and
rejects tuning or `--resume`. It has Word2Vec settings rather than PyTorch
optimizer/AMP/accumulation options. Deterministic execution requires one worker;
`training.num_workers` controls Word2Vec, not DataLoader. Node2Vec supports positive
weighted edges and p/q transitions; DeepWalk uses binary edges and optional restart.
DeepWalk preserves LibCity's combined hierarchical-softmax and negative-sampling
default; set `training.loss.negative_samples=0` for hierarchical softmax alone.
Walks stream to `training/walks.txt`, with JSONL progress and a training report.
Tensor-only weights preserve node/graph/configuration identity; unsafe Gensim model
pickles are not used. Exported vectors follow `node_ids.json`. Reported coverage,
norms and isolated/dead-end node counts are structural diagnostics, not downstream
prediction scores; isolated-node vectors have no context-learning signal. These
small examples use two epochs, while the model protocol retains the legacy
1000-epoch default; use an explicit epoch budget for actual experiments.

STSGCN can be generated with `--model STSGCN`; use the resulting stsgcn.json
with the same train/evaluate/resume commands. The small example enables the
learned graph mask and uses two synchronous blocks; the default model keeps
four blocks. Each block consumes two history frames, so input_window must be
greater than twice the number of blocks. Filters within one block must share
the same width. Both individual and sharing modules, GLU/relu and positional
embedding switches are supported. The preset uses inverse-distance edges and
raw-scale Huber loss with `training.loss.delta=1`; select `masked_huber` explicitly
to exclude zero targets. Mask products are recomputed each forward pass, so
ordinary backward and the public resume lifecycle work without retain_graph.

Generate STPGCN with `--model STPGCN`, then use stpgcn.json with the same public
train/evaluate/resume commands. C/L/d configure channels, layer count and position
width; alpha bounds outgoing spatial hops and beta bounds the causal past.
Both bounds may be zero. Historical UTC weekday and time-slot indices come from
actual source timestamps as int64 history_features, with Monday=0 and slots from
midnight. The sampling interval must be an integer divisor of one day. Calendar
channels do not need to be appended to X. All Gaussian/graph/selection modules
are registered and checkpointed; equivalent factored relations avoid constructing
one dense spatiotemporal graph per query. This remains a dense spatial model.

Generate DGCN with `--model DGCN`; the hourly fixture spans enough history for
two weekly chunks. Run dgcn.json with the same lifecycle commands. Its small
example uses horizon=1 and five CPT frames. The named dgcn_cpt protocol retains
the original five twelve-frame chunks by default, requiring at least twelve
samples per hour under the current common CPT adapter. CPT categories, widths,
odd temporal kernels and attention-head counts are configurable. The corrected
coarse attention groups complete history chunks instead of interleaving their
positions. Training MSE includes raw zero targets by default; metric masks are
configured independently. Numerical checks explicitly account for the axis fix.

The complete supported-model inventory and migration details are in the
[runtime guide](../docs/modernization/runtime/README_zh.md). Small examples prove
execution and reproducibility, not real-city benchmark accuracy.
