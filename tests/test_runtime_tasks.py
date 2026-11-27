import json
import numpy as np
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.executor.train import train_loop
from libcity.model.runtime_registry import model_spec,resolve
from libcity.pipeline.experiment import run_experiment
from runtime_fixtures import task_fixture,atomic_fixture
from runtime_helpers import make_context,tiny_config


def configuration(root,name,exp='train'):
    params={'RNN':dict(hidden_size=8,loc_emb_size=4,tim_emb_size=3,dropout_p=.1),
            'DeepTTE':dict(hidden_size=8,num_filter=4,final_fc_size=8,num_final_fcs=1),
            'LINE':dict(embedding_dim=4)}[name]
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache')),
        model=dict(name=name,params=params),training=dict(epochs=2,batch_size=3,shuffle=True),
        runtime=dict(device='cpu',seed=17,exp_id=exp,output_dir=str(root/'runs')))


@pytest.mark.parametrize('name',['RNN','DeepTTE','LINE'])
def test_atomic_task_train_independent_evaluate_and_label_free_forward(tmp_path,name,monkeypatch):
    task_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path,name)
    context=make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'inspect'}})
    batch=next(iter(context.bundle.test))
    inputs=context.loss_fn.inputs(batch)
    assert not {'y','y_raw','local_y_raw'}&inputs.keys()
    context.model.eval()
    prediction=context.model(inputs)
    assert isinstance(prediction,torch.Tensor) and torch.isfinite(prediction).all()
    assert not any(hasattr(module,key) for module in context.model.modules() for key in ('device','scaler','_scaler','_logger'))
    context.tracker.close()
    result=run_experiment('train',config=cfg)
    # Evaluation must not instantiate an optimizer or a training metric writer.
    import libcity.pipeline.experiment as pipeline
    def forbidden(*a,**kw):
        raise AssertionError('Evaluation attempted training construction')
    monkeypatch.setattr(pipeline,'build_context',forbidden)
    independent=run_experiment('evaluate',config=cfg,checkpoint=result.artifacts['best'],overrides={'runtime':{'exp_id':'eval'}})
    for metric,value in independent.metrics.items():
        assert value==pytest.approx(result.metrics[metric],rel=1e-7)
    assert not (tmp_path/'runs/eval/metrics.jsonl').exists()
    if name=='RNN':
        records=json.loads(independent.artifacts['predictions'].read_text())
        assert records and all(len(row['prediction'])==8 for row in records)
        assert context.config['training']['direction']=='max'
        best=torch.load(result.artifacts['last'],weights_only=True)
        assert best['early_stopper']['best']==max(json.loads(line)['val_Recall@1'] for line in (tmp_path/'runs/train/metrics.jsonl').read_text().splitlines())
    elif name=='DeepTTE':
        records=json.loads(independent.artifacts['predictions'].read_text())
        assert records and all(row['truth_seconds']>100 for row in records)
        assert context.bundle.split_summary['duration_unit']=='seconds'
    else:
        embedding=np.load(independent.artifacts['embeddings'],allow_pickle=False)
        assert embedding.shape==(8,4) and np.isfinite(embedding).all()
        assert json.loads(independent.artifacts['node_ids'].read_text())==list(range(10,18))


@pytest.mark.parametrize('name',['RNN','DeepTTE','LINE'])
def test_task_resume_restores_parameters_lr_and_shuffle(tmp_path,name):
    task_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path,name,'full')
    cfg['training']['scheduler']=dict(name='step',interval='epoch',step_size=1,gamma=.5)
    def train(context,limit=None):
        spec=model_spec(name)
        train_loop(context,resolve(spec.step)(context.loss_fn) if spec.step else None,until_epoch=limit)
        snapshot=torch.load(context.directory/'checkpoints/last.pt',weights_only=True)
        context.tracker.close()
        return snapshot
    full=train(make_context(cfg))
    half=make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'half'}})
    train(half,1)
    resumed=make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'resumed'}})
    resumed.checkpoints.load_last(half.directory/'checkpoints/last.pt',resumed)
    actual=train(resumed)
    for key,value in full['model'].items():
        torch.testing.assert_close(value,actual['model'][key],rtol=0,atol=0)
    assert full['scheduler']==actual['scheduler'] and full['train_state']==actual['train_state']
    assert full['optimizer']['param_groups']==actual['optimizer']['param_groups']


@pytest.mark.parametrize('name',['FNN','GWNET'])
def test_standard_models_match_reference_and_use_default_loop(tmp_path,name):
    atomic_fixture(tmp_path/'raw')
    cfg=tiny_config(tmp_path,epochs=1)
    cfg['model']=dict(name=name,params=dict(hidden_size=4) if name=='FNN' else dict(nhid=4,blocks=2,layers=1,dropout=0.,aptonly=False))
    cfg['data']['input_window']=2
    context=make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'inspect'}})
    if name=='FNN':
        from libcity.model.traffic_speed_prediction.FNN import FNN as Legacy
        legacy=Legacy(dict(hidden_size=4,input_window=2,output_window=2),dict(num_nodes=3,feature_dim=1,output_dim=1))
    else:
        from libcity.model.traffic_speed_prediction.GWNET import GWNET as Legacy
        legacy=Legacy(dict(nhid=4,blocks=2,layers=1,dropout=0.,aptonly=False,apt_layer=False,
            input_window=2,output_window=2,device=torch.device('cpu')),dict(num_nodes=3,feature_dim=1,output_dim=1,adj_mx=context.bundle.metadata.adjacency.copy()))
        # Existing GWNET mistakenly uses Conv1d with 2D kernels on 4D inputs.
        for modules in (legacy.gate_convs,legacy.residual_convs,legacy.skip_convs):
            for index,old in enumerate(modules):
                fixed=torch.nn.Conv2d(old.in_channels,old.out_channels,old.kernel_size,dilation=old.dilation)
                fixed.load_state_dict(old.state_dict())
                modules[index]=fixed
    state=legacy.state_dict()
    loaded=context.model.load_state_dict(state,strict=False)
    assert not loaded.unexpected_keys
    batch=next(iter(context.bundle.train))
    context.model.eval();legacy.eval()
    torch.testing.assert_close(context.model({'X':batch['X']}),legacy({'X':batch['X']}),rtol=1e-5,atol=1e-6)
    assert model_spec(name).train is None and model_spec(name).step is None
    if name=='GWNET':
        assert 'support_0' in dict(context.model.named_buffers())
    context.tracker.close()
    result=run_experiment('train',config=cfg)
    assert result.artifacts['last'].is_file() and np.isfinite(result.metrics['loss'])


@pytest.mark.parametrize('cell',['RNN','LSTM','GRU'])
def test_trajectory_cell_variants_and_batch_one(tmp_path,cell):
    task_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path,'RNN',cell)
    cfg['model']['params']['rnn_type']=cell
    cfg['training'].update(epochs=1,batch_size=1)
    cfg['evaluation']=dict(batch_size=1)
    result=run_experiment('train',config=cfg)
    assert np.isfinite(result.metrics['loss']) and result.artifacts['best'].is_file()


def test_line_first_order_matches_original_dot_product_and_objective(tmp_path):
    task_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path,'LINE','first-order')
    cfg['model']['params']['order']='first'
    context=make_context(cfg)
    batch=next(iter(context.bundle.train))
    from libcity.model.road_representation.LINE import LINE_FIRST
    reference=LINE_FIRST(8,4)
    reference.load_state_dict(context.model.embed.state_dict())
    prediction=context.model(context.loss_fn.inputs(batch))
    torch.testing.assert_close(prediction,reference(batch['I'],batch['J']))
    loss=-torch.nn.functional.logsigmoid(prediction*batch['y']).mean()
    torch.testing.assert_close(context.loss_fn(prediction,batch['y']).loss,loss)
    context.tracker.close()


def test_task_metrics_use_global_counts_with_short_batches_and_seconds():
    from types import SimpleNamespace
    from libcity.data.bundle import DatasetMetadata
    from libcity.executor.task_evaluation import RankingLoss,ETALoss,evaluate_task
    from libcity.utils.normalization import StandardScaler
    class Fixed(torch.nn.Module):
        def forward(self,batch):
            return batch['score']
    metadata=DatasetMetadata('math',info={'location_ids':[10,20,30]},input_layout={'kind':'padded_trajectory'})
    loss=RankingLoss(SimpleNamespace(metadata=metadata),{'name':'nll'})
    prediction=torch.log_softmax(torch.tensor([[9.,8.,7.],[7.,8.,9.],[8.,9.,7.]]),dim=1)
    target=torch.tensor([0,0,1])
    loader=[{'score':prediction[:2],'y':target[:2]},{'score':prediction[2:],'y':target[2:]}]
    result=evaluate_task(Fixed(),loader,device=torch.device('cpu'),loss_fn=loss,
        evaluation=dict(metrics=['Recall@1','Recall@3','MRR@3'],topk=[1,3],save_predictions=False))
    assert result.metrics['Recall@1']==pytest.approx(2/3)
    assert result.metrics['Recall@3']==1.
    assert result.metrics['MRR@3']==pytest.approx((1+1/3+1)/3)
    metadata=DatasetMetadata('math',input_layout={'kind':'planned_route'})
    loss=ETALoss(SimpleNamespace(metadata=metadata,scaler=StandardScaler(100.,10.)),{'name':'mae'})
    loader=[dict(score=torch.tensor([[0.],[1.]]),y=torch.tensor([[0.],[0.]]),y_raw=torch.tensor([[100.],[100.]])),
            dict(score=torch.tensor([[2.]]),y=torch.tensor([[0.]]),y_raw=torch.tensor([[100.]]))]
    result=evaluate_task(Fixed(),loader,device=torch.device('cpu'),loss_fn=loss,
        evaluation=dict(metrics=['MAE','RMSE','MAPE'],save_predictions=False))
    assert result.metrics['MAE']==10.
    assert result.metrics['RMSE']==pytest.approx((500/3)**.5)
    assert result.metrics['MAPE']==pytest.approx(.1)
