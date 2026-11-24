"""Inventory registered components using source metadata, without importing models."""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path


AUDIT_DIR = Path(__file__).resolve().parent
ROOT = AUDIT_DIR.parents[2]
sys.path.insert(0, str(ROOT))
from libcity.registry import COMPONENT_MODULES, component_module
from baseline_audit import OPTIONAL_EXTRAS


def find_component(kind, name, *, task=None):
    try:
        return component_module(kind, name, task=task)
    except AttributeError:
        return None


def external_imports(module, visited=None):
    visited = set() if visited is None else visited
    if module in visited:
        return set()
    visited.add(module)
    path = ROOT.joinpath(*module.split('.')).with_suffix('.py')
    if not path.exists():
        return set()
    result = set()
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            # Executors import Ray only in the hyperparameter tuning branch.
            if node.module == 'ray' and node not in tree.body and [alias.name for alias in node.names] == ['tune']:
                continue
            names = [node.module] + [node.module + '.' + n.name for n in node.names]
        else:
            continue
        for name in names:
            if name.startswith('libcity.'):
                result.update(external_imports(name, visited))
            else:
                result.add(name.split('.')[0])
    return result


def inventory():
    task_file = ROOT / 'libcity/config/task_config.json'
    tasks = json.loads(task_file.read_text(encoding='utf-8'))
    for task, settings in tasks.items():
        for model in settings['allowed_model']:
            name = 'RNN' if model in ('RNN', 'GRU', 'LSTM') else model
            module = find_component('model', name, task=task)
            spec = settings.get(model)
            issues = []
            modules = [module] if module else []
            if module is None:
                issues.append('model registration missing')
            if spec is None:
                issues.append('task component mapping missing')
            else:
                for key in COMPONENT_MODULES:
                    if key not in spec:
                        if key in ('dataset_class', 'executor', 'evaluator'):
                            issues.append(key + ' mapping missing')
                        continue
                    target = find_component(key, spec[key])
                    if target:
                        modules.append(target)
                    else:
                        issues.append(key + ' registration missing: ' + spec[key])
                for field, folder in [('dataset_class', 'data'), ('executor', 'executor'), ('evaluator', 'evaluator')]:
                    if field in spec and not (ROOT / 'libcity/config' / folder / (spec[field] + '.json')).exists():
                        issues.append(folder + ' config missing: ' + spec[field])
            if not (ROOT / 'libcity/config/model' / task / (name + '.json')).exists():
                issues.append('model config missing')
            dependencies = set().union(*(external_imports(m) for m in modules))
            extras = sorted({OPTIONAL_EXTRAS[d] for d in dependencies if d in OPTIONAL_EXTRAS})
            yield {'task': task, 'model': model, 'module': module or '',
                   'optional_extras': ';'.join(extras),
                   'status': 'registration_incomplete' if issues else 'compatibility_pending',
                   'validation': 'static_inventory_only', 'issues': '; '.join(issues)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=AUDIT_DIR / 'model_compatibility.csv')
    parser.add_argument('--runtime-report', type=Path,
                        help='Merge config and import results from entrypoint_audit.py.')
    args = parser.parse_args()
    rows = list(inventory())
    if args.runtime_report:
        with args.runtime_report.open(encoding='utf-8') as handle:
            runtime = {(row['task'], row['model']): row for row in csv.DictReader(handle)}
        for row in rows:
            record = runtime[(row['task'], row['model'])]
            row['validation'] = 'config_and_import_passed' if record['status'] == 'import_passed' else 'config_passed_import_attempted'
            row['status'] = record['status']
            row['issues'] = '; '.join(filter(None, (row['issues'], record['issues'])))
            if row['model'] in ('TemplateTLP', 'TemplateTSP'):
                row['status'] = 'template_only'
                row['issues'] = '; '.join(filter(None, (row['issues'], 'Developer template, not an implemented trainable model')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} task/model entries to {args.output}')


if __name__ == '__main__':
    main()
