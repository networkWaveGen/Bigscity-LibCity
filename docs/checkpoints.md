# Runtime checkpoints

The runtime accepts **.pt** files only, loaded with `weights_only=True` onto CPU
before copying values into a model on its selected device. There is no unsafe
unpickling fallback, old Executor checkpoint loader or migration command.

| File | Contents | Use |
| --- | --- | --- |
| best.pt | Pure model state_dict, including registered buffers | `run_model.py evaluate --checkpoint ...` |
| last.pt | Model, optimizer, scheduler, AMP, training/task progress, early stopping, best weights, random states and data contract | `run_model.py train --resume ...` |

Resume validates model, training settings, dataset metadata, input layout and
normalization state. Inference weights do not silently reset optimizer state.
The target-time split contract introduced in phase three is checked on resume.

Historical `.m`, `.tar`, object-pickle, tuple-pair and epoch-dictionary loaders
have been removed. Historical formats are documented in Git history, not an
alternate active loading API. See the [source lifecycle examples](../examples/README.md)
and [runtime guide](modernization/runtime/README_zh.md).
