"""Archive upstream registrations and audit dependencies and existing results.

Uses Git source text and installed distribution metadata, never file hashes.
Run from the validated core environment: python docs/modernization/audit/baseline_audit.py.
"""

import argparse
import ast
import csv
from collections import deque
from importlib import metadata
from importlib.util import resolve_name
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


AUDIT_DIR = Path(__file__).resolve().parent
ROOT = AUDIT_DIR.parents[2]
BASELINE = '5a6391d41944e937f2c15e9be85ab7f40ac8b23e'

# Import names used to classify optional dependencies in audit reports.
OPTIONAL_EXTRAS = {
    'ray': 'tune', 'hyperopt': 'tune', 'bayes_opt': 'tune',
    'dgl': 'graph', 'fastdtw': 'dtw', 'gensim': 'embeddings',
    'torchtext': 'geosan', 'nltk': 'geosan',
    'einops': 'transformer', 'infomap': 'multispans',
    'torch_geometric': 'multispans', 'pywt': 'wavelet', 'tslearn': 'pdformer',
    'torchdiffeq': 'ode', 'statsmodels': 'statistics',
}


def git_text(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8-sig')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def exports(source, package):
    """Read package-level re-exports from Git or the working tree."""
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module != '__future__':
            module = resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
            for alias in node.names:
                result[alias.asname or alias.name] = (module, alias.name)
    return result


def archive_registrations(output):
    paths = git_text('ls-tree', '-r', '--name-only', BASELINE, 'libcity').splitlines()
    rows, changes, baseline_exports = [], [], {}
    for path in paths:
        if not path.endswith('/__init__.py'):
            continue
        package = path[:-12].replace('/', '.')
        before = exports(git_text('show', BASELINE + ':' + path), package)
        if not before:
            continue
        baseline_exports[package] = before
        after = exports((ROOT / path).read_text(encoding='utf-8-sig'), package)
        for name, (module, attribute) in before.items():
            rows.append(dict(package=package, name=name, module=module, attribute=attribute))
        changes.append(dict(package=package, before_count=len(before), after_count=len(after),
                            removed=sorted(before.keys() - after.keys()),
                            added=sorted(after.keys() - before.keys()),
                            redirected={name: dict(before=before[name], after=after[name])
                                        for name in sorted(before.keys() & after.keys()) if before[name] != after[name]}))
    with (output / 'baseline_components.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['package', 'name', 'module', 'attribute'])
        writer.writeheader()
        writer.writerows(rows)
    tasks = json.loads(git_text('show', BASELINE + ':libcity/config/task_config.json'))
    write_json(output / 'baseline_task_config.json', tasks)
    model_rows = []
    current_tasks = json.loads((ROOT / 'libcity/config/task_config.json').read_text(encoding='utf-8'))
    task_changes = []
    for task, settings in tasks.items():
        for name in settings['allowed_model']:
            real_name = 'RNN' if name in ('GRU', 'LSTM') else name
            targets = [values[real_name][0] for package, values in baseline_exports.items()
                       if package.startswith('libcity.model.') and real_name in values]
            if task == 'traj_loc_pred':
                targets = [target for target in targets if '.trajectory_loc_prediction.' in target]
            elif task == 'traffic_state_pred':
                targets = [target for target in targets if '.trajectory_loc_prediction.' not in target]
            targets = sorted(set(targets))
            spec = settings.get(name, {})
            model_rows.append(dict(task=task, model=name, module=';'.join(targets),
                                   dataset=spec.get('dataset_class', ''), executor=spec.get('executor', ''),
                                   evaluator=spec.get('evaluator', ''),
                                   encoder=spec.get('traj_encoder', spec.get('eta_encoder', '')),
                                   issues='' if spec else 'baseline task mapping missing'))
            after = current_tasks.get(task, {}).get(name)
            if settings.get(name) != after:
                task_changes.append(dict(task=task, model=name, before=settings.get(name), after=after))
    with (output / 'baseline_models.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)
    write_json(output / 'task_registration_changes.json', dict(baseline=BASELINE, changes=task_changes,
        removed_allowed={task: sorted(set(settings['allowed_model']) -
                                     set(current_tasks.get(task, {}).get('allowed_model', [])))
                         for task, settings in tasks.items()}))
    write_json(output / 'registration_changes.json', dict(baseline=BASELINE, packages=changes))
    return dict(public_components=len(rows), packages=len(changes),
                task_model_registrations=sum(len(task['allowed_model']) for task in tasks.values()),
                removed_exports=sum(len(change['removed']) for change in changes))


def core_dependency_paths(requirements):
    """Follow active, installed Requires-Dist edges, including requested extras."""
    paths, visited = {}, set()
    pending = deque((Requirement(item), ['libcity core']) for item in requirements)
    missing = set()
    while pending:
        requirement, chain = pending.popleft()
        name = canonicalize_name(requirement.name)
        paths.setdefault(name, chain + [name])
        key = (name, tuple(sorted(requirement.extras)))
        if key in visited:
            continue
        visited.add(key)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.add(name)
            continue
        for text in distribution.requires or []:
            child = Requirement(text)
            if child.marker is None or any(child.marker.evaluate({'extra': extra})
                                           for extra in ('', *requirement.extras)):
                pending.append((child, paths[name]))
    return paths, sorted(missing)


def dependency_audit(output):
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    old = [Requirement(line) for line in git_text('show', f'{BASELINE}:requirements.txt').splitlines()
           if line.strip() and not line.startswith('#')]
    core = {canonicalize_name(Requirement(item).name) for item in project['dependencies']}
    groups = {}
    for extra, requirements in project['optional-dependencies'].items():
        for text in requirements:
            groups.setdefault(canonicalize_name(Requirement(text).name), []).append(extra)
    imports = {}
    sources = list((ROOT / 'libcity').rglob('*.py')) + list(ROOT.glob('*.py'))
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
            for name in names:
                imports.setdefault(name.split('.')[0].lower(), set()).add(path.relative_to(ROOT).as_posix())
    aliases = {'scikit-learn': 'sklearn', 'ray': 'ray', 'tensorboardx': 'tensorboardx',
               'pep8-naming': 'pep8ext_naming'}
    graph, missing = core_dependency_paths(project['dependencies'])
    rows = []
    for requirement in old:
        name = canonicalize_name(requirement.name)
        uses = sorted(imports.get(aliases.get(name, name.replace('-', '_')), []))
        if name in core:
            disposition = 'modern_core'
        elif name in groups:
            disposition = 'optional:' + ','.join(groups[name])
        elif name in graph:
            disposition = 'transitive_core_no_direct_pin'
        elif name in ('flake8', 'pep8-naming'):
            disposition = 'legacy_lint_tool_not_runtime'
        elif uses:
            disposition = 'unclassified_direct_use'
        else:
            disposition = 'not_used_by_core_or_source'
        rows.append(dict(requirement=str(requirement), disposition=disposition,
                         direct_source_uses=uses, core_dependency_path=graph.get(name, [])))
    write_json(output / 'dependency_audit.json', dict(
        environment=sys.version.split()[0], baseline=BASELINE,
        method='All project Python imports plus active installed Requires-Dist closure from modern core.',
        scope='Runtime core on this platform. Optional stacks retain their own transitive requirements; '
              'an indirect optional dependency need not remain a direct core pin. Not a cross-platform lock.',
        missing_core_distributions=missing, requirements=rows))
    return dict(legacy_requirements=len(rows), missing_core_distributions=missing,
                unclassified=[row['requirement'] for row in rows if row['disposition'] == 'unclassified_direct_use'])


def history_audit(output):
    workspace = ROOT.parent
    roots = [workspace / 'LibCity的github代码', workspace / 'LibCity2.0-main' / 'LibCity',
             ROOT / 'libcity/cache', ROOT / 'libcity/log']
    reports, matches = [], []
    for root in roots:
        examined = 0
        if root.exists():
            for directory, folders, files in os.walk(root):
                folders[:] = sorted(name for name in folders if name not in (
                    '.git', '.venv', 'node_modules', '__pycache__', 'AgentCity'))
                for name in sorted(files):
                    examined += 1
                    path = Path(directory) / name
                    suffix = path.suffix.lower()
                    location = '/'.join(part.lower() for part in path.relative_to(root).parts)
                    artifact = suffix in ('.log', '.pt', '.pth', '.m', '.tar', '.npz')
                    result = suffix in ('.csv', '.json', '.txt') and any(
                        word in location for word in ('evaluate_cache', 'model_cache', 'result', 'metrics'))
                    if artifact or result:
                        matches.append(dict(path=str(path.relative_to(workspace)), bytes=path.stat().st_size,
                                            kind='candidate_existing_experiment_artifact'))
        reports.append(dict(path=str(root.relative_to(workspace)), exists=root.exists(), files_examined=examined))
    write_json(output / 'historical_experiments.json', dict(
        searched=reports, existing_candidates=matches,
        conclusion='No existing historical experiment artifacts found in searched LibCity directories.' if not matches
                   else 'Candidates listed; inspect provenance before treating these as accuracy baselines.',
        exclusions='AgentCity excluded as agreed. .runtime-tests contains newly generated synthetic checks, '
                   'not historical real-data experiments. No historical results were fabricated.',
        historical_accuracy_baseline_available=False))
    return dict(existing_experiment_candidates=len(matches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=AUDIT_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(registrations=archive_registrations(args.output),
                   dependencies=dependency_audit(args.output), history=history_audit(args.output))
    write_json(args.output / 'baseline_audit_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))
    if summary['registrations']['removed_exports'] or summary['dependencies']['unclassified'] or summary['dependencies']['missing_core_distributions']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
