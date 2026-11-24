"""Check explicit component imports and task registrations in the core environment.

Optional stacks may be absent; that is recorded distinctly from broken core
imports. Import resolution is not a claim that every model has been trained.
"""

import argparse
import ast
from collections import Counter
import csv
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess

AUDIT_DIR = Path(__file__).resolve().parent
ROOT = AUDIT_DIR.parents[2]
sys.path.insert(0, str(ROOT))
from libcity.config.config_parser import ConfigParser
from libcity.registry import COMPONENT_MODULES, load_component
from baseline_audit import OPTIONAL_EXTRAS, write_json


MOVED_IMPLEMENTATIONS = {
    ('libcity.pipeline.pipeline', 'hyper_parameter'): 'libcity.pipeline.tuning',
    ('libcity.utils.utils', 'get_executor'): 'libcity.registry',
    ('libcity.utils.utils', 'get_model'): 'libcity.registry',
    ('libcity.utils.utils', 'get_evaluator'): 'libcity.registry',
    ('libcity.utils.utils', 'get_logger'): 'libcity.utils.experiment',
    ('libcity.utils.utils', 'get_local_time'): 'libcity.utils.experiment',
    ('libcity.utils.utils', 'set_random_seed'): 'libcity.utils.experiment',
    ('libcity.utils.utils', 'ensure_dir'): 'libcity.utils.filesystem',
    ('libcity.utils.utils', 'preprocess_data'): 'libcity.utils.dataset',
    ('libcity.utils.utils', 'trans_naming_rule'): None,
}


def checked_import(module, name):
    try:
        getattr(import_module(module), name)
        return 'import_passed', ''
    except ModuleNotFoundError as error:
        missing = error.name
        if missing in OPTIONAL_EXTRAS:
            return 'optional_dependency_missing', f'{missing}: install extra {OPTIONAL_EXTRAS[missing]}'
        return 'error', f'{type(error).__name__}: {error}'
    except Exception as error:
        return 'error', f'{type(error).__name__}: {error}'


def audit_implementations():
    with (AUDIT_DIR / 'baseline_components.csv').open(encoding='utf-8') as handle:
        baseline = list(csv.DictReader(handle))
    rows = []
    for entry in baseline:
        name = entry['attribute']
        module = MOVED_IMPLEMENTATIONS.get((entry['module'], name), entry['module'])
        if module is None:
            status, issue = 'removed', 'No in-repository callers'
        else:
            status, issue = checked_import(module, name)
        rows.append(dict(**entry, implementation_module=module or '', status=status, issues=issue))
    return rows


def audit_registrations(work_dir):
    tasks = json.loads((ROOT / 'libcity/config/task_config.json').read_text(encoding='utf-8'))
    rows = []
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='registrations-', dir=work_dir) as directory:
        try:
            os.chdir(directory)
            fixture = Path('raw_data/audit')
            fixture.mkdir(parents=True)
            (fixture / 'config.json').write_text('{}', encoding='utf-8')
            for task, settings in tasks.items():
                for model in settings['allowed_model']:
                    row = dict(task=task, model=model, status='import_passed', issues='', checked_components='')
                    statuses, issues, checked = [], [], []
                    try:
                        config = ConfigParser(task, model, 'audit', other_args={'gpu': False})
                        components = [('model', config['model'])]
                        components += [(field, config[field]) for field in COMPONENT_MODULES if field in config]
                        for field, name in components:
                            try:
                                component = load_component(field, name, task=task)
                                assert callable(component), f'{field} is not callable'
                                statuses.append('import_passed')
                                checked.append(field + '=' + name)
                            except ModuleNotFoundError as error:
                                missing = error.name
                                if missing not in OPTIONAL_EXTRAS:
                                    raise
                                statuses.append('optional_dependency_missing')
                                issues.append(f'{field}={name}: missing {missing} ({OPTIONAL_EXTRAS[missing]})')
                    except Exception as error:
                        statuses.append('error')
                        issues.append(f'{type(error).__name__}: {error}')
                    row['status'] = 'error' if 'error' in statuses else (
                        'optional_dependency_missing' if 'optional_dependency_missing' in statuses else 'import_passed')
                    row['issues'] = '; '.join(issues)
                    row['checked_components'] = '; '.join(checked)
                    rows.append(row)
        finally:
            os.chdir(previous)
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_script_imports():
    """Exercise script imports without running training code at module scope."""
    rows = []
    scripts = [*ROOT.glob('*.py'), *(ROOT / 'test').glob('*.py')]
    for path in sorted(scripts):
        for node in ast.parse(path.read_text(encoding='utf-8-sig')).body:
            if not isinstance(node, ast.ImportFrom) or not (node.module or '').startswith('libcity.'):
                continue
            for alias in node.names:
                status, issue = checked_import(node.module, alias.name)
                rows.append(dict(script=path.relative_to(ROOT).as_posix(), module=node.module, name=alias.name,
                                 status=status, issues=issue))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=AUDIT_DIR)
    parser.add_argument('--work-dir', type=Path, default=AUDIT_DIR)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    if not work_dir.is_relative_to(ROOT.parent.resolve()):
        parser.error('--work-dir must be inside the LibCity workspace')
    work_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    implementations = audit_implementations()
    registrations = audit_registrations(work_dir)
    scripts = audit_script_imports()
    write_csv(args.output / 'implementation_imports.csv', implementations)
    write_csv(args.output / 'registration_imports.csv', registrations)
    write_csv(args.output / 'script_imports.csv', scripts)
    cli = {}
    for name in ('run_model.py', 'visualize.py'):
        result = subprocess.run([sys.executable, '-B', str(ROOT / name), '--help'],
                                cwd=ROOT, capture_output=True, text=True, errors='replace')
        cli[name] = dict(status='passed' if result.returncode == 0 else 'error', returncode=result.returncode)
    syntax_errors = []
    for path in (ROOT / 'libcity').rglob('*.py'):
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'))
        except SyntaxError as error:
            syntax_errors.append(f'{path.relative_to(ROOT)}: {error}')
    summary = dict(implementation_imports=dict(Counter(row['status'] for row in implementations)),
                   registrations=dict(Counter(row['status'] for row in registrations)),
                   script_imports=dict(Counter(row['status'] for row in scripts)), cli_help=cli,
                   tasks=sorted({row['task'] for row in registrations}), syntax_errors=syntax_errors,
                   templates=['traj_loc_pred/TemplateTLP', 'traffic_state_pred/TemplateTSP'],
                   scope='Explicit implementation imports, config loading and component resolution. '
                         'Optional dependency absence is isolated; model training is separately validated.')
    write_json(args.output / 'entrypoint_audit_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))
    for row in implementations + registrations + scripts:
        if row['status'] == 'error':
            print(json.dumps(row, ensure_ascii=False))
    if syntax_errors or any(row['status'] == 'error' for row in implementations + registrations + scripts) or any(
            entry['status'] == 'error' for entry in cli.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
