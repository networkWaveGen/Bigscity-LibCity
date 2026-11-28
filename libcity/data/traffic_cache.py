"""Reuse unscaled atomic traffic arrays without caching split-dependent windows."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np


FIELDS = ('history', 'target', 'known', 'clock', 'intervals', 'graph', 'known_indices')


def source_files(data, metadata):
    folder = Path(data['data_dir']) / data['dataset']
    files = [folder / 'config.json']
    for key, suffix in (('geo_file', '.geo'), ('rel_file', '.rel'), ('data_files', '.dyna')):
        if key=='rel_file' and data['graph']['name']=='none':
            continue
        names = metadata.get(key, data['dataset'])
        names = [names] if isinstance(names, str) else names
        files.extend(folder / (name + suffix) for name in names)
    if data['features']['load_external']:
        files.append(folder / (metadata.get('ext_file', data['dataset']) + '.ext'))
    return files


def source_signature(paths):
    """Use file metadata for invalidation; never hash source file contents."""
    result = []
    for path in paths:
        stat = path.stat() if path.exists() else None
        result.append(dict(path=str(path.resolve()), size=stat.st_size if stat else None,
                           mtime_ns=stat.st_mtime_ns if stat else None))
    return result


def prepare_arrays(data, metadata, build, logger=None, *, namespace=None, source_paths=None, fields=FIELDS, validate=None):
    if not data['cache']:
        return build(), dict(status='disabled', path=None)
    parameters = dict(version=1, root=str((Path(data['data_dir']) / data['dataset']).resolve()),
                      features=data['features'], graph=data['graph'])
    if namespace is not None:
        parameters['adapter'] = namespace
    # This hashes configuration text only, matching the existing cache naming convention.
    key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:24]
    folder = Path(data['cache_dir']) / 'traffic-raw-v1'
    path = folder / (key + '.npz')
    sources = source_files(data, metadata) if source_paths is None else source_paths
    identity = dict(parameters=parameters, sources=source_signature(sources))
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as archive:
                if json.loads(str(archive['identity'].item())) == identity:
                    arrays = {name: archive[name] for name in fields}
                    if fields == FIELDS and (arrays['history'].shape[:2] != arrays['target'].shape[:2]
                            or arrays['known'].shape[:2] != arrays['target'].shape[:2]
                            or len(arrays['clock']) != len(arrays['target'])):
                        raise ValueError('Inconsistent cached traffic array dimensions')
                    if validate is not None:
                        validate(arrays)
                    if logger:
                        logger.info('Reused atomic traffic cache: %s', path)
                    return arrays, dict(status='hit', path=str(path.resolve()))
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            if logger:
                logger.warning('Rebuilding unreadable traffic cache %s: %s', path, error)
    arrays = build()
    if validate is not None:
        validate(arrays)
    if source_signature(sources) != identity['sources']:
        raise ValueError('Atomic source files changed during preprocessing; retry with stable inputs')
    folder.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, identity=np.asarray(json.dumps(identity, sort_keys=True)), **arrays)
        # Readers see either the previous complete file or this complete replacement.
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if logger:
        logger.info('Saved atomic traffic cache: %s', path)
    return arrays, dict(status='miss', path=str(path.resolve()))
