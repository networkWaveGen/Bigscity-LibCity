"""Pretrained semantic means, original valid computations and native lifecycle."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.data.semantic_trajectory import load_vectors
from libcity.data.sources import atomic_sources
from libcity.executor.train import train_loop
from libcity.model.trajectory_loc_prediction.SERM import SERM as ReferenceSERM
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import serm_fixture
from runtime_helpers import make_context


def configuration(root,experiment):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='SERM',params=dict(loc_emb_size=4,tim_emb_size=2,hidden_size=4)),
        training=dict(epochs=2,batch_size=8,shuffle=True,accumulation_steps=2,
            scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


def test_pretrained_vectors_are_intact_trainable_and_means_preserve_duplicates_and_empty_categories(tmp_path):
    serm_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'pretrained'))
    try:
        model,info = context.model,context.bundle.metadata.info
        torch.testing.assert_close(model.emb_text.weight,torch.tensor(info['word_vectors'].copy()),rtol=0,atol=0)
        assert model.emb_text.weight.requires_grad and not hasattr(model,'word_one_hot_matrix')
        assert len(info['word_vocabulary'])==5 and context.bundle.split_summary['semantic_empty_pois']==2
        expected = torch.tensor([[0.,1.,2.5],[.5,.25,.75],[1.,2.,3.],[0.,0.,0.],[0.,0.,0.],
            [-1.,0.,2.],[.25,.625,-.125],[0.,1.,-1.]])
        actual = model.semantics(torch.arange(1,9)[None])[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actual.square().sum().backward()
        assert model.emb_text.weight.grad[1:].abs().sum()>0
        assert (model.emb_text.weight.grad[0]==0).all()
        assert 'poi_words' in dict(model.named_buffers())
        assert all(not hasattr(module,'device') and not hasattr(module,'_scaler') for module in model.modules())
    finally:
        context.tracker.close()


class ReferenceBatch(dict):
    def get_origin_len(self,key):
        return self['lengths'].tolist()


def test_original_one_hot_mean_classifier_and_all_gradients_with_same_weights(tmp_path):
    serm_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'reference'))
    try:
        model,info = context.model,context.bundle.metadata.info
        feature = {**info,'text_size':len(info['word_vectors'])-1,'word_vec':info['word_vectors'][1:].tolist()}
        reference = ReferenceSERM({**context.config['model']['params'],'device':torch.device('cpu')},feature)
        weights = reference.state_dict()
        missing = model.load_state_dict({name:value for name,value in weights.items() if not name.startswith('emb_text.')},strict=False)
        assert set(missing.missing_keys)=={'emb_text.weight','poi_words'} and not missing.unexpected_keys
        with torch.no_grad():
            model.emb_text.weight[1:].copy_(reference.emb_text.layer.weight.T)
        batch = next(iter(context.bundle.train))
        text = []
        for row,length in zip(batch['current_loc'],batch['lengths']):
            text.append([[int(word)-1 for word in info['poi_word_ids'][int(location)] if word]
                for location in row[:int(length)]])
        legacy = ReferenceBatch(**batch,text=text)
        previous = reference(legacy)[:,1:]
        expected = previous-torch.logsumexp(previous,dim=-1,keepdim=True)
        actual = model(context.loss_fn.inputs(batch))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        F.nll_loss(expected,batch['y']).backward()
        F.nll_loss(actual,batch['y']).backward()
        parameters = dict(reference.named_parameters())
        for name,parameter in model.named_parameters():
            if name=='emb_text.weight':
                torch.testing.assert_close(parameter.grad[1:],reference.emb_text.layer.weight.grad.T,rtol=2e-4,atol=1e-6)
                assert (parameter.grad[0]==0).all()
            else:
                torch.testing.assert_close(parameter.grad,parameters[name].grad,rtol=2e-4,atol=1e-6)
    finally:
        context.tracker.close()


def test_complete_training_ranking_export_and_exact_rmsprop_resume(tmp_path):
    serm_fixture(tmp_path/'raw')
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial'))
    train_loop(partial,until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state'] and expected['scheduler']==actual['scheduler']
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)
    assert expected['optimizer']['param_groups']==actual['optimizer']['param_groups']
    for identity,state in expected['optimizer']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(value,actual['optimizer']['state'][identity][key],rtol=0,atol=0)
    records = json.loads(evaluated.artifacts['predictions'].read_text())
    assert records and all(len(row['prediction'])==8 and row['truth'] in range(10,18) for row in records)


def test_padding_batch_independence_float64_label_free_and_owned_inputs(tmp_path):
    serm_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'padding'))
    try:
        model = context.model.double().eval()
        batch = next(iter(context.bundle.train))
        inputs = context.loss_fn.inputs(batch)
        before = {key:value.clone() for key,value in inputs.items()}
        prediction = model(inputs)
        for index in range(len(batch['y'])):
            one = {key:value[index:index+1] for key,value in inputs.items()}
            torch.testing.assert_close(model(one),prediction[index:index+1],rtol=1e-12,atol=1e-12)
        valid = torch.arange(inputs['current_loc'].shape[1])[None]<inputs['lengths'][:,None]
        poisoned = {key:value.clone() for key,value in inputs.items()}
        poisoned['current_loc'][~valid] = 999999
        poisoned['current_tim'][~valid] = -999
        poisoned['y'] = torch.full_like(batch['y'],9999)
        rng = torch.get_rng_state().clone()
        torch.testing.assert_close(model(poisoned),prediction,rtol=0,atol=0)
        assert prediction.shape==(len(batch['y']),8) and prediction.dtype==torch.float64
        torch.testing.assert_close(prediction.exp().sum(-1),torch.ones(len(prediction),dtype=torch.float64))
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key,value in before.items():
            torch.testing.assert_close(value,inputs[key],rtol=0,atol=0)
        context.loss_fn(prediction,batch['y']).loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        with pytest.raises(ValueError,match='lengths'):
            model({**inputs,'lengths':torch.zeros_like(inputs['lengths'])})
    finally:
        context.tracker.close()


def test_semantics_align_real_geo_ids_after_reordering_and_use_metadata_relative_path(tmp_path,monkeypatch):
    folder = serm_fixture(tmp_path/'raw')
    geo = pd.read_csv(folder/'TaskTiny.geo').iloc[[7,3,1,6,0,4,2,5]]
    geo.to_csv(folder/'TaskTiny.geo',index=False)
    monkeypatch.chdir(tmp_path.parent)
    context = make_context(configuration(tmp_path,'real-ids'))
    try:
        info = context.bundle.metadata.info
        identities = list(info['location_ids'])
        assert identities[0]==17 and info['semantic_column']=='venue_category_name'
        # Original geo_id=10 is token 5 after reordering, rather than iloc[10].
        location = identities.index(10)+1
        expected = torch.tensor([[[0.,1.,2.5]]])
        torch.testing.assert_close(context.model.semantics(torch.tensor([[location]])),expected,rtol=0,atol=0)
        assert context.bundle.split_summary['semantic_known_words']==4
    finally:
        context.tracker.close()


def test_vector_reader_keeps_only_requested_words_and_stops_before_unneeded_tail(tmp_path):
    path = tmp_path/'vectors.txt'
    path.write_text('\ncoffee 1 2 3\nunused not-a-vector\n',encoding='utf-8')
    selected,dimension = load_vectors(path,{'coffee'})
    assert set(selected)=={'coffee'} and dimension==3
    np.testing.assert_array_equal(selected['coffee'],[1,2,3])
    # Absent words require scanning to EOF, but unrelated vectors are not parsed.
    selected,dimension = load_vectors(path,{'coffee','not-in-glove'})
    assert set(selected)=={'coffee'} and dimension==3


@pytest.mark.parametrize('contents',['','\nunused bad 1\n','unused 0 0 0\ncoffee 1 2\n','unused 0 0 0\ncoffee nan 1 2\n'])
def test_bad_pretrained_vectors_fail_at_data_entry(tmp_path,contents):
    path = tmp_path/'vectors.txt';path.write_text(contents,encoding='utf-8')
    with pytest.raises(ValueError,match='SERM word vector file'):
        load_vectors(path,{'coffee'})


@pytest.mark.parametrize('missing',['word_vector_file','semantic_column','resource'])
def test_missing_semantic_resources_have_explicit_errors(tmp_path,missing):
    folder = serm_fixture(tmp_path/'raw')
    path = folder/'config.json'
    metadata = json.loads(path.read_text())
    if missing=='resource':
        metadata['word_vector_file'] = 'absent.txt'
    elif missing=='semantic_column':
        metadata['semantic_column'] = 'absent_column'
    else:
        metadata.pop('word_vector_file')
    path.write_text(json.dumps(metadata))
    with pytest.raises((ValueError,FileNotFoundError),match='SERM'):
        make_context(configuration(tmp_path,'missing'))


def test_all_unknown_or_missing_categories_have_zero_semantics_and_train(tmp_path):
    folder = serm_fixture(tmp_path/'raw')
    geo = pd.read_csv(folder/'TaskTiny.geo')
    geo['venue_category_name'] = ['not-in-glove',None]*4
    geo.to_csv(folder/'TaskTiny.geo',index=False)
    context = make_context(configuration(tmp_path,'empty'))
    try:
        assert context.bundle.split_summary['semantic_known_words']==0
        assert context.bundle.split_summary['semantic_empty_pois']==8
        actual = context.model.semantics(torch.arange(1,9)[None])
        assert actual.shape==(1,8,3) and (actual==0).all()
        batch = next(iter(context.bundle.train))
        prediction = context.model(context.loss_fn.inputs(batch))
        context.loss_fn(prediction,batch['y']).loss.backward()
        assert torch.isfinite(prediction).all()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in context.model.parameters())
    finally:
        context.tracker.close()


def test_external_pretrained_file_version_change_rejects_resume_before_optimizer_update(tmp_path,monkeypatch):
    folder = serm_fixture(tmp_path/'raw')
    first = run_experiment('train',config={**configuration(tmp_path,'first'),
        'training':{**configuration(tmp_path,'first')['training'],'epochs':1}})
    resource = folder.parent/'word_vec/glove.fixture.txt'
    sources = atomic_sources(dict(data_dir=str(tmp_path/'raw'),dataset='TaskTiny'))
    assert str(resource.resolve()) in {row['name'] for row in sources}
    # Keep all used vectors unchanged: the resource version guard must still fire.
    with resource.open('a',encoding='utf-8') as stream:
        stream.write('new-unused-word 9 8 7\n')
    def forbidden(*args,**kwargs):
        pytest.fail('Changed pretrained sources must fail before the next update')
    monkeypatch.setattr(torch.optim.RMSprop,'step',forbidden)
    with pytest.raises(ValueError,match='atomic source files changed'):
        run_experiment('train',config=configuration(tmp_path,'changed'),resume=first.artifacts['last'])


def test_actual_cpu_amp_training(tmp_path):
    serm_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0
