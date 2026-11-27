import numpy as np
import torch
from libcity.executor.stssl_train import train,dynamic_weights
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from runtime_helpers import make_context,tiny_config


def configuration(root,name):
    cfg=tiny_config(root,name,epochs=3)
    cfg['data'].update(input_window=12,output_window=1)
    cfg['model']=dict(name='STSSL',params=dict(d_model=8,dropout=.1,nmb_prototype=3,percent=.2))
    cfg['training']['scheduler']=dict(name='step',interval='epoch',step_size=1,gamma=.7)
    return cfg


def test_stssl_dynamic_task_state_next_updates_and_rng_resume(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=100,nodes=6)
    full=make_context(configuration(tmp_path,'full'))
    train(full)
    expected=torch.load(full.directory/'checkpoints/last.pt',weights_only=True)
    full.tracker.close()
    half=make_context(configuration(tmp_path,'half'))
    train(half,until_epoch=2)
    half.tracker.close()
    resumed=make_context(configuration(tmp_path,'resume'))
    resumed.checkpoints.load_last(half.directory/'checkpoints/last.pt',resumed)
    train(resumed)
    actual=torch.load(resumed.directory/'checkpoints/last.pt',weights_only=True)
    resumed.tracker.close()
    assert expected['train_state']==actual['train_state']
    assert expected['scheduler']==actual['scheduler']
    assert actual['train_state']['task_state']['dynamic_weighting_epoch']==3
    task=actual['train_state']['task_state']
    assert len(task['previous_losses'])==3 and np.isfinite(task['loss_weights']).all()
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)
    weights=dynamic_weights([3.,2.,1.],[1.,2.,3.],2.)
    assert np.isclose(sum(weights),3.) and weights[0]>weights[1]>weights[2]


def test_stssl_prediction_has_no_augmentation_or_labels_and_pipeline_exports(tmp_path,monkeypatch):
    atomic_fixture(tmp_path/'raw',steps=80,nodes=6)
    context=make_context(configuration(tmp_path,'inspect'))
    context.model.eval()
    batch=next(iter(context.bundle.test))
    before=torch.get_rng_state().clone()
    first=context.model({'X':batch['X']})
    torch.testing.assert_close(first,context.model({'X':batch['X']}),rtol=0,atol=0)
    torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
    assert 'graph' in context.model.state_dict()
    context.tracker.close()
    cfg=configuration(tmp_path,'pipeline')
    cfg['training']['epochs']=1
    result=run_experiment('train',config=cfg)
    evaluated=run_experiment('evaluate',config=cfg,checkpoint=result.artifacts['best'],overrides={'runtime':{'exp_id':'eval'}})
    assert evaluated.metrics['loss']==result.metrics['loss']
    assert evaluated.artifacts['predictions'].is_file()
