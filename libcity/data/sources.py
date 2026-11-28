"""Portable atomic source versions, based on filesystem metadata, never content hashes."""
import json
from pathlib import Path


SUFFIXES = {'.geo','.rel','.dyna','.grid','.od','.gridod','.ext','.usr'}
FILE_FIELDS = {'geo_file':'.geo','rel_file':'.rel','data_files':'.dyna','dyna_file':'.dyna',
               'ext_file':'.ext','usr_file':'.usr','truth_file':'.dyna','flow_files':'.gridod'}


def atomic_sources(data):
    folder = (Path(data['data_dir'])/data['dataset']).resolve()
    metadata_file = folder/'config.json'
    raw = json.loads(metadata_file.read_text(encoding='utf-8'))
    metadata = {**raw,**raw.get('info',{})}
    paths = {metadata_file}
    paths.update(path.resolve() for path in folder.rglob('*') if path.is_file() and path.suffix.lower() in SUFFIXES)
    # Explicit references may point outside the dataset directory or name an
    # auxiliary format. Keep these references rather than just scanning siblings.
    for key,value in metadata.items():
        if key not in FILE_FIELDS and not key.endswith(('_file','_files')):
            continue
        values = [value] if isinstance(value,str) else value if isinstance(value,list) else []
        for name in values:
            if isinstance(name,str):
                suffix = FILE_FIELDS.get(key,'')
                if (key=='data_files' and data.get('sampling',{}).get('name') in ('grid_cpt','grid_window','risk_cpt','stdn_grid')
                        and (folder/(name+'.grid')).is_file()):
                    suffix = '.grid'
                if key=='data_files' and data.get('sampling',{}).get('name')=='od_window':
                    suffix = next((candidate for candidate in ('.od','.gridod','.dyna')
                        if (folder/(name+candidate)).is_file()),'.od')
                paths.add((folder/(name+suffix)).resolve())
    records = []
    for path in paths:
        name = path.relative_to(folder).as_posix() if path.is_relative_to(folder) else str(path)
        stat = path.stat() if path.is_file() else None
        records.append(dict(name=name,size=stat.st_size if stat else None,mtime_ns=stat.st_mtime_ns if stat else None))
    return sorted(records,key=lambda row:row['name'])
