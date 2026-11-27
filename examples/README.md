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
The complete supported-model inventory and migration details are in the
[runtime guide](../docs/modernization/runtime/README_zh.md). Small examples prove
execution and reproducibility, not real-city benchmark accuracy.
