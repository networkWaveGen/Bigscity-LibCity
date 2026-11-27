# Historical reference material

These flat JSON presets and classical baseline functions are retained for future
model migration and numerical research. They are excluded from the installed
libcity package and are not runnable entrypoints for the new experiment runtime.
The old standalone baseline `__main__` blocks have been removed.

Current users should start with [the source examples](../examples/README.md).
The supported model list is the [runtime migration inventory](../docs/modernization/runtime/model-migration.csv).
Old ConfigParser, custom Batch objects, Executor classes and checkpoint migration
utilities have been deleted; this directory does not reintroduce them.
