"""Historical attention, causal session banks and the common ranking lifecycle."""
import json

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.data.task_bundles import collate_sequences
from libcity.executor.train import train_loop
from libcity.model.trajectory_loc_prediction.DeepMove import DeepMove as ReferenceDeepMove
from libcity.model.trajectory_loc_prediction.runtime_deepmove import Attention
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import task_fixture
from runtime_helpers import make_context


def configuration(root,experiment,rnn='LSTM',attention='dot'):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='DeepMove',params=dict(loc_emb_size=4,tim_emb_size=2,hidden_size=4,
            rnn_type=rnn,attn_type=attention,dropout_p=.2)),
        training=dict(epochs=2,batch_size=8,shuffle=True,scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


class ReferenceInput(dict):
    def get_origin_len(self,key):
        return self['lengths' if key=='current_loc' else 'history_lengths'].tolist()


@pytest.mark.parametrize('rnn',['GRU','LSTM'])
@pytest.mark.parametrize('attention',['dot','general'])
def test_original_nonpadded_recurrent_attention_and_gradients(tmp_path,rnn,attention):
    task_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'reference',rnn,attention))
    try:
        params = context.config['model']['params']
        reference = ReferenceDeepMove(dict(params,device=torch.device('cpu'),evaluate_method='all'),dict(context.bundle.metadata.info)).eval()
        model = context.model.eval()
        keys = model.state_dict().keys()
        weights = reference.state_dict()
        assert set(weights)-set(keys)=={'fc_attn.weight','fc_attn.bias'}
        model.load_state_dict({key:value for key,value in weights.items() if key in keys})
        batch = ReferenceInput(current_loc=torch.tensor([[1,2,3],[3,4,5]]),current_tim=torch.tensor([[1,2,3],[4,5,6]]),
            lengths=torch.tensor([3,3]),history_loc=torch.tensor([[2,3,4,5],[5,6,7,8]]),
            history_tim=torch.tensor([[7,8,9,10],[11,12,13,14]]),history_lengths=torch.tensor([4,4]))
        old = reference(batch)
        expected = F.log_softmax(old[:,1:],dim=-1)
        actual = model(batch)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(actual.exp().sum(-1),torch.ones(2))
        F.nll_loss(expected,torch.tensor([1,3])).backward()
        F.nll_loss(actual,torch.tensor([1,3])).backward()
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-5,atol=1e-6)
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger')) for module in model.modules())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('rnn,attention',[('GRU','dot'),('LSTM','general'),('RNN','concat')])
def test_complete_training_ranking_export_and_exact_dropout_resume(tmp_path,rnn,attention):
    task_fixture(tmp_path/'raw')
    cfg = lambda name: configuration(tmp_path,name,rnn,attention)
    full = run_experiment('train',config=cfg('full'))
    evaluated = run_experiment('evaluate',config=cfg('evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(cfg('partial'))
    train_loop(partial,until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=cfg('resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert actual['train_state']==expected['train_state']
    assert actual['scheduler']==expected['scheduler']
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)
    records = json.loads(evaluated.artifacts['predictions'].read_text())
    assert records and all(len(row['prediction'])==8 and row['truth'] in range(10,18) for row in records)


@pytest.mark.parametrize('empty_history',[False,True])
def test_padding_batch_independence_float64_and_empty_history(tmp_path,empty_history):
    task_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'padding',attention='general'))
    try:
        model = context.model.eval().double()
        batch = dict(current_loc=torch.tensor([[1,2,0],[3,4,5]]),current_tim=torch.tensor([[1,2,0],[3,4,5]]),
            lengths=torch.tensor([2,3]),history_loc=torch.tensor([[2,0,0,0],[4,5,6,7]]),
            history_tim=torch.tensor([[1,0,0,0],[2,3,4,5]]),history_lengths=torch.tensor([0 if empty_history else 1,4]))
        before = {key:value.clone() for key,value in batch.items()}
        rng = torch.get_rng_state()
        prediction = model(batch)
        individual = {key:value[:1] for key,value in batch.items()}
        individual['current_loc'],individual['current_tim'] = individual['current_loc'][:,:2],individual['current_tim'][:,:2]
        stop = 0 if empty_history else 1
        individual['history_loc'],individual['history_tim'] = individual['history_loc'][:,:stop],individual['history_tim'][:,:stop]
        torch.testing.assert_close(prediction[:1],model(individual),rtol=1e-6,atol=1e-7)
        extended = {key:F.pad(value,(0,3),value=9999) if value.ndim==2 else value for key,value in individual.items()}
        extended['y'] = torch.full((1,),9999)
        torch.testing.assert_close(prediction[:1],model(extended),rtol=1e-6,atol=1e-7)
        assert prediction.shape==(2,8) and prediction.dtype==torch.float64 and torch.isfinite(prediction).all()
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key,value in before.items():
            torch.testing.assert_close(value,batch[key],rtol=0,atol=0)
        prediction.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    finally:
        context.tracker.close()


def test_concat_attention_against_explicit_scores_and_all_empty_mask():
    attention = Attention('concat',4).double()
    query = torch.randn(2,4,dtype=torch.float64)
    history = torch.randn(2,3,4,dtype=torch.float64)
    valid = torch.tensor([[True,True,False],[False,False,False]])
    scores = torch.stack([torch.stack([attention.other@torch.tanh(attention.attn(torch.cat((query[b],history[b,t]))))
        for t in range(3)]) for b in range(2)])
    expected = torch.zeros_like(scores)
    expected[0,:2] = scores[0,:2].softmax(0)
    actual = attention(query,history,valid)
    torch.testing.assert_close(actual,expected)
    assert torch.isfinite(actual).all() and not actual[1].any()


def test_history_is_causal_and_benchmark_samples_are_preserved(tmp_path):
    folder = task_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'history'))
    base_config = configuration(tmp_path,'baseline')
    base_config['model'] = dict(name='RNN',params={'hidden_size':4})
    baseline = make_context(base_config)
    try:
        for name in ('train','val','test'):
            rows = getattr(context.bundle,name).dataset.rows
            original = getattr(baseline.bundle,name).dataset.rows
            assert {int(row['sample_id']) for row in rows}=={int(row['sample_id']) for row in original}
        assert sum(context.bundle.split_summary['cold_start_samples'].values())==5
        frame = pd.read_csv(folder/'TaskTiny.dyna')
        frame['time'] = pd.to_datetime(frame['time'],utc=True)
        groups = [part.sort_values('time') for _,part in frame.groupby(['entity_id','traj_id'])]
        loc_ids = {identity:index+1 for index,identity in enumerate(context.bundle.metadata.info['location_ids'])}
        for loader in (context.bundle.train,context.bundle.val,context.bundle.test):
            for row in loader.dataset.rows:
                target = frame[frame.dyna_id==int(row['sample_id'])].iloc[0]
                trip = frame[(frame.entity_id==target.entity_id)&(frame.traj_id==target.traj_id)]
                previous = [part for part in groups if part.entity_id.iloc[0]==target.entity_id and part.time.max()<trip.time.min()]
                previous.sort(key=lambda part:part.time.max())
                expected = [loc_ids[identity] for part in previous for identity in part.location]
                assert row['history_loc'].tolist()==expected
                assert int(row['history_lengths'])==len(expected)
        examples = [row for row in context.bundle.train.dataset.rows if int(row['history_lengths'])>0][:2]
        before = [row['history_loc'].clone() for row in examples]
        collated = collate_sequences(examples);collated['history_loc'].fill_(9999)
        for row,value in zip(examples,before):
            torch.testing.assert_close(row['history_loc'],value,rtol=0,atol=0)
    finally:
        context.tracker.close();baseline.tracker.close()
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    mask = frame.traj_id==0
    frame.loc[frame[mask].index[-1],'time'] = '2024-01-03T00:30:00Z'
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    overlap = make_context(configuration(tmp_path,'overlap'))
    try:
        current_ids = set(frame.loc[frame.traj_id==2,'dyna_id'])
        rows = [row for loader in (overlap.bundle.train,overlap.bundle.val,overlap.bundle.test)
                for row in loader.dataset.rows if int(row['sample_id']) in current_ids]
        assert rows and all(int(row['history_lengths'])==0 for row in rows)
    finally:
        overlap.tracker.close()


@pytest.mark.parametrize('rnn',['RNN','GRU','LSTM'])
def test_actual_cpu_amp_training(tmp_path,rnn):
    task_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'amp',rnn,'concat')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0
