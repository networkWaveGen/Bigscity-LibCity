"""Prove source/data read-only execution in a generated snapshot, never the checkout.

Windows uses a temporary inherited deny-Write/Delete ACL for the current identity.
POSIX uses read/execute modes. Child probes must prove writes fail; flags alone
are not accepted as evidence. Original checkout/data permissions are untouched.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    directory=args.directory.resolve()
    if directory.is_relative_to(repo):
        raise ValueError('Use a new snapshot directory outside the checkout')
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'.runtime-readonly-sandbox').write_text('generated validation snapshot only\n')
    source=directory/'source'
    source.mkdir()
    names=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode('utf-8').split('\0')
    for name in filter(None,names):
        original=repo/name
        destination=source/name
        if not destination.resolve().is_relative_to(source):
            raise ValueError('Snapshot path escapes its destination')
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,destination)
    work=directory/'work'
    work.mkdir()
    commands=[]
    environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(source),
                 'PYTHONIOENCODING':'utf-8'}
    def run(arguments,*,env=None):
        result=subprocess.run([sys.executable,'-B',*map(str,arguments)],cwd=work,
            env=env or environment,capture_output=True,text=True,encoding='utf-8',errors='replace')
        commands.append(dict(arguments=list(map(str,arguments)),exit_code=result.returncode))
        if result.returncode:
            raise RuntimeError(result.stdout+'\n'+result.stderr)
        return result.stdout
    bootstrap=directory/'bootstrap'
    run([source/'tools/make_runtime_fixture.py','--directory',bootstrap])
    data=directory/'data'
    shutil.copytree(bootstrap/'data',data)
    config=json.loads((bootstrap/'gru.json').read_text())
    config['data'].update(data_dir=str(data),cache_dir=str(directory/'cache'))
    config['runtime']['output_dir']=str(directory/'runs')
    config_file=work/'gru.json'
    config_file.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    protected=[]
    posix_modes={}
    def permissions(path,action):
        if path not in (source,data) or not path.is_relative_to(directory):
            raise ValueError('Permission target is not a generated snapshot')
        if os.name=='nt':
            shell=shutil.which('pwsh.exe') or 'powershell.exe'
            result=subprocess.run([shell,'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
                '-File',str(repo/'tools/readonly_acl.ps1'),'-Action',action,'-TargetPath',str(path)],
                capture_output=True,text=True,encoding='utf-8',errors='replace')
            if result.returncode:
                raise RuntimeError(result.stderr)
        elif action=='protect':
            for item in [path,*path.rglob('*')]:
                posix_modes[item]=item.stat().st_mode&0o777
                item.chmod(0o555 if item.is_dir() else 0o444)
        else:
            for item,mode in posix_modes.items():
                if item==path or item.is_relative_to(path):
                    item.chmod(mode)
    report=dict(passed=False,python=platform.python_version(),platform=platform.platform(),
        directory=str(directory),source=str(source),data=str(data),cwd=str(work),commands=commands,
        protection='inherited Windows deny Write/Delete ACL' if os.name=='nt' else 'POSIX permissions')
    try:
        for path in (source,data):
            permissions(path,'protect')
            protected.append(path)
        probes=[source/'write-probe',source/'run_model.py',data/'write-probe',data/'Tiny/config.json']
        code='''import json,sys
from pathlib import Path
for name in json.loads(sys.argv[1]):
    try:
        with Path(name).open('ab'): pass
    except PermissionError: continue
    raise AssertionError('Write access was not blocked: '+name)
print('all writes blocked')'''
        assert 'all writes blocked' in run(['-c',code,json.dumps(list(map(str,probes)))])
        report['write_probes_denied']=len(probes)
        location=run(['-c','import libcity;print(libcity.__file__)']).strip()
        assert Path(location).resolve().is_relative_to(source)
        report['actual_import_path']=location
        junit=directory/'pytest.xml'
        output=run(['-m','pytest','-q','--maxfail=3','--tb=short',source/'tests',
                    '--basetemp',directory/'pytest-temp','--junitxml',junit,
                    '-o','cache_dir='+str(directory/'cache/pytest')])
        suite=ET.parse(junit).getroot().find('testsuite')
        assert all(int(suite.get(key))==0 for key in ('failures','errors','skipped'))
        report['full_tests']=dict(tests=int(suite.get('tests')),failures=0,errors=0,skipped=0,
            seconds=float(suite.get('time')),junit=str(junit),
            names=[item.get('classname')+'::'+item.get('name') for item in suite.iter('testcase')])
        def lifecycle(mode,*extra):
            return json.loads(run([source/'run_model.py',mode,'--config',config_file,*extra]).strip().splitlines()[-1])
        trained=lifecycle('train','--set','runtime.exp_id=train')
        evaluated=lifecycle('evaluate','--checkpoint',trained['artifacts']['best'],'--set','runtime.exp_id=evaluate')
        resumed=lifecycle('train','--resume',trained['artifacts']['last'],'--set','training.epochs=3','--set','runtime.exp_id=resumed')
        for key,value in evaluated['metrics'].items():
            assert abs(value-trained['metrics'][key])<1e-7
        import torch
        state=torch.load(resumed['artifacts']['last'],weights_only=True)
        assert state['train_state']['epoch']==3
        assert not (directory/'runs/evaluate/metrics.jsonl').exists()
        report.update(passed=True,training=trained['metrics'],evaluation=evaluated['metrics'],resumed_epoch=3,
            cuda_available=torch.cuda.is_available(),mps_available=torch.backends.mps.is_available(),
            environment=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines())
    finally:
        for path in reversed(protected):
            permissions(path,'restore')
        report['permissions_restored']=True
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,tests=report['full_tests']['tests'],report=str(args.report))))


if __name__=='__main__':
    main()
