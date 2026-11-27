import csv
import importlib.util
import json
from pathlib import Path
from libcity.model.runtime_registry import MODELS


def test_migration_inventory_covers_every_original_task_model_and_truthful_capabilities():
    repo=Path(__file__).resolve().parents[1]
    task_config=json.loads((repo/'libcity/config/task_config.json').read_text())
    expected={(task,name) for task,config in task_config.items() for name in config['allowed_model']}
    rows=list(csv.DictReader((repo/'docs/modernization/runtime/model-migration.csv').open(encoding='utf-8-sig')))
    assert {(row['task'],row['model']) for row in rows}==expected
    assert len(rows)==len(expected)==100
    for row in rows:
        if row['migration_status']=='migrated':
            spec=MODELS[row['runtime_name']]
            assert spec.task==row['task']
            assert row['requires_checkpoint']==str(spec.requires_checkpoint)
            assert row['supports_resume']==str(spec.supports_resume)
            assert row['supports_validation']==str(spec.supports_validation)
            assert 'not tested' not in row['actual_validation']
        else:
            assert row['actual_validation']=='not tested in new runtime'
            assert row['supports_resume']=='unverified'
