"""Actual representative CUDA training/evaluation/resume; no CPU fallback."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from libcity.testing.fixtures import atomic_fixture
from libcity.pipeline.experiment import run_experiment
from libcity.model.runtime_registry import model_spec,resolve


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA validation requires an available CUDA PyTorch build; no fallback')
    directory=args.directory.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    atomic_fixture(directory/'data',steps=100,nodes=6)
    records={}
    for name,params in [('GRU',dict(hidden_size=8)),
                        ('STGCN',dict(Kt=2,Ks=2,blocks=[[1,4,4],[4,4,4]])),
                        ('DCRNN',dict(rnn_units=8,num_rnn_layers=1,max_diffusion_step=1))]:
        observed={}
        cls=resolve(model_spec(name).factory)
        def observe(module,inputs):
            if type(module) is cls:
                assert all(value.device.type=='cuda' for value in module.parameters())
                assert all(value.device.type=='cuda' for value in module.buffers())
                assert inputs[0]['X'].device.type=='cuda'
                observed.update(parameters='cuda',inputs='cuda',buffers={key:str(value.device) for key,value in module.named_buffers()})
        handle=torch.nn.modules.module.register_module_forward_pre_hook(observe)
        config=dict(data=dict(dataset='Tiny',data_dir=str(directory/'data'),cache_dir=str(directory/'cache'),
            cache=False,pad_training=False,input_window=12,output_window=2),model=dict(name=name,params=params),
            training=dict(epochs=2,batch_size=8,amp=name=='GRU',amp_dtype='float16'),
            runtime=dict(device='cuda:0',seed=17,exp_id=name+'-train',output_dir=str(directory/'runs')))
        try:
            trained=run_experiment('train',config=config)
            evaluated=run_experiment('evaluate',config=config,checkpoint=trained.artifacts['best'],
                overrides={'runtime':{'exp_id':name+'-evaluate'}})
            for key,value in evaluated.metrics.items():
                assert np.isclose(value,trained.metrics[key],rtol=1e-5,atol=1e-6)
            resumed=run_experiment('train',config=config,resume=trained.artifacts['last'],
                overrides={'runtime':{'exp_id':name+'-resumed'},'training':{'epochs':3}})
            state=torch.load(resumed.artifacts['last'],map_location='cpu',weights_only=True)
            assert state['train_state']['epoch']==3 and state['train_state']['optimizer_steps']>0
            assert observed
            if name=='GRU':
                assert state['amp_scaler']
                full=run_experiment('train',config=config,
                    overrides={'runtime':{'exp_id':name+'-full'},'training':{'epochs':3}})
                reference=torch.load(full.artifacts['last'],map_location='cpu',weights_only=True)
                for key,value in reference['model'].items():
                    torch.testing.assert_close(value,state['model'][key],rtol=1e-5,atol=1e-6)
                assert reference['train_state']==state['train_state']
                assert reference['amp_scaler']==state['amp_scaler']
            records[name]=dict(passed=True,metrics=evaluated.metrics,observed_devices=observed,
                resumed_epoch=3,optimizer_steps=state['train_state']['optimizer_steps'],
                amp_scaler=state['amp_scaler'],amp_resume_matches_continuous=name=='GRU')
        finally:
            handle.remove()
    report=dict(passed=True,python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
        cuda_runtime=torch.version.cuda,device=torch.cuda.get_device_name(0),models=records,
        environment='Isolated venv inheriting existing CUDA torch; fresh CPU environment is validated separately')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,device=report['device'],models=list(records),report=str(args.report))))


if __name__=='__main__':
    main()
