"""Geographic words, reference attention, bounded sampling and safe lifecycles."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.config.runtime import load_config
from libcity.data.dataset.dataset_subclass.geosan_dataset import latlon2quadkey
from libcity.data.quadkeys import QuadkeyVocabulary,quadkey,quadkey_ngrams
from libcity.data.task_bundles import collate_sequences,prepare_personalized_trajectory
from libcity.data.trajectory_points import spherical_distance
from libcity.executor.geosan_step import GeographicSampler,sampled_step
from libcity.executor.train import train_loop
from libcity.model.trajectory_loc_prediction.GeoSAN import GeoSAN as ReferenceGeoSAN
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import strnn_fixture,task_fixture
from runtime_helpers import make_context


def configuration(root,name,**params):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='GeoSAN',params=dict(location_dim=4,region_dim=4,num_heads=2,num_layers=1,
            dropout=0.,max_len=32,candidate_chunk_size=4,**params)),
        training=dict(epochs=2,batch_size=8,shuffle=True,
            scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


def test_quadkeys_match_legacy_geometry_and_frozen_field_vocabulary_rules():
    # TorchText 0.6: Field(unk_token=None) -> PAD=0; freq desc then alphabetical.
    vocabulary = QuadkeyVocabulary()
    vocabulary.build_vocab([['222','111','222'],['333','111'],['444']])
    assert vocabulary.vocab.itos==['<pad>','111','222','333','444']
    assert vocabulary.numericalize([['222','111'],['333']]).tolist()==[[2,1],[3,0]]
    vocabulary.build_vocab([['222','111','222'],['333','111']],min_freq=2)
    assert vocabulary.itos==['<pad>','111','222']
    with pytest.raises(KeyError): vocabulary.numericalize([['unknown']])
    for longitude,latitude in [(116.4,39.9),(0,0),(-73.98,40.76),(180,90),(-180,-90),(179.9,-85.05112878)]:
        for level in (1,6,17,23):
            assert quadkey(longitude,latitude,level)==latlon2quadkey(latitude,longitude,level)
    assert quadkey(0,0,1)=='3'
    assert quadkey_ngrams('01230123',6)==['012301','123012','230123']


@pytest.mark.parametrize('attention,position',[(False,'transformer'),(True,'transformer'),(False,'embedding'),(True,'embedding')])
def test_original_geographic_encoder_decoder_scores_and_all_gradients(tmp_path,attention,position):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'reference',use_attention_as_decoder=attention,position_encoding=position))
    try:
        model,info = context.model.eval(),context.bundle.metadata.info
        reference = ReferenceGeoSAN(dict(device=torch.device('cpu'),executor_config={'train':{'num_negative_samples':3,'temperature':1.}},
            model_config=dict(user_embedding_dim=8 if attention else 0,location_embedding_dim=4,time_embedding_dim=0,
                region_embedding_dim=4,num_heads_encoder=2,num_layers_encoder=1,dropout=0.,extra_config=dict(
                    embedding_fusion='concat',user_embedding=False,position_encoding=position,
                    size_sqrt_regularize=True,use_attention_as_decoder=attention))),
            dict(nuser=info['uid_size'],nloc=info['loc_size'],ntime=169,nquadkey=len(info['quadkey_tokens']))).eval()
        original = reference.state_dict()
        weights = model.state_dict()
        for name in weights:
            if name!='quadkey_codes': weights[name] = original[name][:weights[name].shape[0]]
        model.load_state_dict(weights)
        batch = dict(current_loc=torch.tensor([[1,2,3],[4,5,0]]),current_tim=torch.tensor([[1,2,3],[4,5,0]]),
            lengths=torch.tensor([3,2]),uid=torch.tensor([1,2]))
        candidates = torch.arange(8)
        tokens = (candidates+1)[:,None].expand(-1,2)
        old = reference(batch['uid'][:,None].expand(-1,3).T,batch['current_loc'].T,
            model.quadkey_codes[batch['current_loc']].transpose(0,1),batch['current_tim'].T,
            reference._generate_square_mask_(3,torch.device('cpu')),torch.tensor([[False,False,False],[False,False,True]]),
            tokens,model.quadkey_codes[tokens],None,ds=batch['lengths'].tolist()).T
        actual = model(batch)
        torch.testing.assert_close(actual,old,rtol=3e-5,atol=3e-6)
        actual.square().mean().backward();old.square().mean().backward()
        parameters = dict(reference.named_parameters())
        for name,parameter in model.named_parameters():
            expected = parameters[name].grad
            if expected is not None: expected = expected[:parameter.shape[0]]
            torch.testing.assert_close(parameter.grad,expected,rtol=3e-4,atol=3e-6)
        assert all(not hasattr(module,'device') and not hasattr(module,'_scaler') for module in model.modules())
    finally:
        context.tracker.close()


def test_weighted_binary_loss_importance_weights_and_score_gradients(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'loss'))
    try:
        loss_fn = context.loss_fn
        loss_fn.temperature = .7
        scores = torch.tensor([[.5,-.2,.8,1.4],[-.7,1.,-.4,.2]],dtype=torch.float64,requires_grad=True)
        probabilities = torch.tensor([[.2,.3,.5],[.1,.6,.3]],dtype=torch.float64)
        actual = loss_fn.selected(scores,probabilities).loss
        negative = scores[:,1:]
        weights = torch.softmax(negative/.7-probabilities.log(),-1)
        expected = (-F.logsigmoid(scores[:,0])+(F.softplus(negative)*weights).sum(-1)).mean()
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(actual,scores,retain_graph=True)[0],torch.autograd.grad(expected,scores)[0])
        target = torch.tensor([0,2])
        full = loss_fn(scores,target).loss
        reference = torch.stack([F.softplus(-scores[index,truth])+(F.softplus(torch.cat((scores[index,:truth],scores[index,truth+1:])))
            *torch.softmax(torch.cat((scores[index,:truth],scores[index,truth+1:]))/.7,0)).sum() for index,truth in enumerate(target)]).mean()
        torch.testing.assert_close(full,reference,rtol=1e-12,atol=1e-12)
    finally:
        context.tracker.close()


def test_geographic_sampler_uses_radians_excludes_positive_and_is_bounded():
    coordinates = np.array([[120.,80.],[121.,80.],[120.,79.5],[0.,0.]])
    sampler = GeographicSampler(coordinates,2,[[],[1]],True)
    expected = np.argsort(spherical_distance(coordinates[0],coordinates))[1:3]
    np.testing.assert_array_equal(sampler.neighbours(0),expected)
    negative,probability = sampler.sample(torch.tensor([0]),torch.tensor([1]),100)
    assert negative.unique().tolist()==[2] and probability.unique().tolist()==[1.]
    sampler.visited = [[],[1,2]]
    torch.manual_seed(17)
    negative,probability = sampler.sample(torch.tensor([0]),torch.tensor([1]),6000)
    assert set(negative.flatten().tolist())=={1,2} and abs(float((negative==1).float().mean())-.5)<.04
    assert probability.unique().tolist()==[.5]
    sampler = GeographicSampler(np.zeros((140,2)),500)
    for positive in range(140):
        pool = sampler.neighbours(positive)
        assert len(pool)==139 and positive not in pool
    assert len(sampler.cache)==128
    single = GeographicSampler(np.zeros((1,2)),2000)
    negative,probability = single.sample(torch.tensor([0]),torch.tensor([0]),5)
    assert negative.shape==probability.shape==(1,0)


def test_canonical_samples_static_vocabulary_and_training_only_user_statistics(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'features'))
    try:
        cfg,info = context.config,context.bundle.metadata.info
        base = prepare_personalized_trajectory(cfg['data'],cfg['training'],cfg['evaluation'],17)
        for split in ('train','val','test'):
            assert [int(row['sample_id']) for row in getattr(context.bundle,split).dataset.rows]==[
                int(row['sample_id']) for row in getattr(base,split).dataset.rows]
        frame = pd.read_csv(folder/'TaskTiny.dyna');frame['time'] = pd.to_datetime(frame['time'],utc=True)
        geo = pd.read_csv(folder/'TaskTiny.geo')
        words = [quadkey_ngrams(latlon2quadkey(*json.loads(value)[::-1],17),6) for value in geo.coordinates]
        vocabulary = QuadkeyVocabulary();vocabulary.build_vocab(words)
        assert info['quadkey_tokens']==tuple(vocabulary.itos)
        np.testing.assert_array_equal(info['quadkey_codes'][1:],vocabulary.numericalize(words).numpy())
        cut = pd.Timestamp(context.bundle.split_summary['train_end'],tz='UTC')
        for user,identities in enumerate(info['training_visited'][1:],1):
            expected = frame[(frame.entity_id==info['user_ids'][user])&(frame.time<cut)].location.to_numpy()-10
            assert set(identities)==set(expected)
        row = context.bundle.test.dataset.rows[-1]
        before = {key:value.clone() for key,value in row.items()}
        prediction = context.model(context.loss_fn.inputs(collate_sequences([row])))
        mask = frame.dyna_id==int(row['sample_id'])
        frame.loc[mask,'location'] = 10+(int(frame.loc[mask,'location'].iloc[0])-9)%8
        frame['time'] = frame.time.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        frame.to_csv(folder/'TaskTiny.dyna',index=False)
        changed = make_context(configuration(tmp_path,'changed'))
        try:
            new = next(item for item in changed.bundle.test.dataset.rows if int(item['sample_id'])==int(row['sample_id']))
            assert int(new['y'])!=int(before['y'])
            for key,value in context.loss_fn.inputs(before).items(): torch.testing.assert_close(value,new[key],rtol=0,atol=0)
            assert info['quadkey_tokens']==changed.bundle.metadata.info['quadkey_tokens']
            assert info['training_visited']==changed.bundle.metadata.info['training_visited']
            torch.testing.assert_close(prediction,changed.model(changed.loss_fn.inputs(collate_sequences([new]))),rtol=0,atol=0)
        finally: changed.tracker.close()
    finally: context.tracker.close()


def test_causal_padding_batch_chunks_float64_and_candidate_gradient_equivalence(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'padding',use_attention_as_decoder=True))
    try:
        model = context.model.double().eval()
        inputs = context.loss_fn.inputs(next(iter(context.bundle.train)))
        before = {key:value.clone() for key,value in inputs.items()}
        state = {key:value.clone() for key,value in model.state_dict().items()}
        prediction = model(inputs)
        assert prediction.dtype==torch.float64 and torch.isfinite(prediction).all()
        for index in range(len(prediction)):
            one = {key:value[index:index+1] for key,value in inputs.items()}
            torch.testing.assert_close(model(one),prediction[index:index+1],rtol=1e-11,atol=1e-11)
        valid = torch.arange(inputs['current_loc'].shape[1])[None]<inputs['lengths'][:,None]
        poisoned = {key:value.clone() for key,value in inputs.items()}
        for key in ('current_loc','current_tim'): poisoned[key][~valid] = 999999
        poisoned['y'] = torch.full_like(inputs['uid'],9999)
        torch.testing.assert_close(model(poisoned),prediction,rtol=0,atol=0)
        model.chunk_size = 1
        torch.testing.assert_close(model(inputs),prediction,rtol=1e-11,atol=1e-11)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs),prediction,rtol=1e-11,atol=1e-11)
        selected = torch.tensor([[0,2,2]]).expand(len(prediction),-1)
        actual = model.forward_candidates(inputs,selected)
        expected = prediction.gather(1,selected)
        torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
        variables = list(model.parameters())
        gradients = torch.autograd.grad(actual.square().mean(),variables,retain_graph=True)
        reference = torch.autograd.grad(expected.square().mean(),variables)
        for value,old in zip(gradients,reference): torch.testing.assert_close(value,old,rtol=1e-10,atol=1e-10)
        full = dict(current_loc=torch.tensor([[1,2,3,4]]),current_tim=torch.tensor([[1,2,3,4]]),lengths=torch.tensor([4]),uid=torch.tensor([1]))
        prefix = {key:value[:,:2] if value.ndim==2 else value.clone() for key,value in full.items()};prefix['lengths'][:] = 2
        torch.testing.assert_close(model.encode(full)[0][:,:2],model.encode(prefix)[0],rtol=1e-11,atol=1e-11)
        with torch.no_grad():
            torch.testing.assert_close(model.encode(full)[0][:,:2],model.encode(prefix)[0],rtol=1e-11,atol=1e-11)
        for key,value in before.items(): torch.testing.assert_close(value,inputs[key],rtol=0,atol=0)
        for key,value in state.items(): torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    finally: context.tracker.close()


@pytest.mark.parametrize('fusion,location_only',[('concat',False),('multiply',False),('concat',True)])
def test_corrected_fusion_user_time_odd_width_and_trainable_parameters(tmp_path,fusion,location_only):
    strnn_fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'fusion',embedding_fusion=fusion,use_location_only=location_only)
    cfg['model']['params'].update(location_dim=3,region_dim=3,user_dim=0 if location_only else 3,
        time_dim=0 if location_only else 3,num_heads=1,use_attention_as_decoder=True)
    context = make_context(cfg)
    try:
        batch = next(iter(context.bundle.train))
        result = sampled_step(context.loss_fn)(context.model,batch,context.state)
        result.loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in context.model.parameters())
        assert context.model(context.loss_fn.inputs(batch)).shape==(len(batch['y']),8)
    finally: context.tracker.close()


def test_full_sampled_training_ranking_export_and_exact_dropout_rng_resume(tmp_path,monkeypatch):
    strnn_fixture(tmp_path/'raw')
    def cfg(name):
        value = configuration(tmp_path,name,use_attention_as_decoder=True)
        value['model']['params']['dropout'] = .2
        value['training']['loss'] = dict(exclude_visited=True)
        return value
    full = run_experiment('train',config=cfg('full'))
    with monkeypatch.context() as guard:
        def reject(*args,**kwargs): raise AssertionError('Evaluation must not build the training KNN index')
        guard.setattr('libcity.executor.geosan_step.GeographicSampler',reject)
        evaluation = run_experiment('evaluate',config=cfg('evaluation'),checkpoint=full.artifacts['best'])
    assert evaluation.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(cfg('partial'));train_loop(partial,sampled_step(partial.loss_fn),until_epoch=1);partial.tracker.close()
    resumed = run_experiment('train',config=cfg('resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert actual['train_state']==expected['train_state'] and actual['scheduler']==expected['scheduler']
    torch.testing.assert_close(actual['rng']['torch'],expected['rng']['torch'],rtol=0,atol=0)
    for key,value in expected['model'].items(): torch.testing.assert_close(value,actual['model'][key],rtol=0,atol=0)
    records = json.loads(evaluation.artifacts['predictions'].read_text())
    assert records and all(len(row['prediction'])==8 and row['truth'] in range(10,18) for row in records)
    assert json.loads(evaluation.artifacts['runtime_info'].read_text())['benchmark_protocol']=='geosan_ranking'


def test_actual_cpu_amp_and_single_location_positive_only_training(tmp_path):
    folder = task_fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'amp')
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert np.isfinite(list(run_experiment('train',config=cfg).metrics.values())).all()
    geo = pd.read_csv(folder/'TaskTiny.geo').iloc[:1];geo.to_csv(folder/'TaskTiny.geo',index=False)
    frame = pd.read_csv(folder/'TaskTiny.dyna');frame['location'] = 10;frame.to_csv(folder/'TaskTiny.dyna',index=False)
    cfg = configuration(tmp_path,'single');cfg['training']['epochs'] = 1
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all() and result.metrics['Recall@1']==1.
    last = torch.load(result.artifacts['last'],weights_only=True)
    assert last['train_state']['optimizer_steps']>0


def test_torchtext_and_nltk_are_unneeded_and_invalid_config_fails(tmp_path):
    script = """
import builtins,sys
original=builtins.__import__
def blocked(name,*args,**kwargs):
    if name.split('.')[0] in {'torchtext','nltk'}: raise ImportError('legacy dependency blocked')
    return original(name,*args,**kwargs)
builtins.__import__=blocked
from libcity.data.dataset.dataset_subclass.geosan_dataset import GeoSANDataset
from libcity.model.trajectory_loc_prediction.runtime_geosan import GeoSAN
from libcity.data.geosan_trajectory import prepare_geosan
from pathlib import Path
from libcity.testing.fixtures import strnn_fixture
from libcity.pipeline.experiment import run_experiment
root=Path(sys.argv[1]);strnn_fixture(root/'raw')
cfg=dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
    model=dict(name='GeoSAN',params=dict(location_dim=2,region_dim=2,num_heads=2,num_layers=1,max_len=32,dropout=0.)),
    training=dict(epochs=1,batch_size=16),runtime=dict(device='cpu',output_dir=str(root/'runs'),exp_id='blocked'))
assert run_experiment('train',config=cfg).metrics['loss']>=0
assert not {'torchtext','nltk'} & set(sys.modules)
"""
    result = subprocess.run([sys.executable,'-B','-c',script,str(tmp_path/'blocked')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    cfg = configuration(tmp_path,'invalid')
    for patch in [dict(data={'quadkey_ngram':18}),dict(training={'loss':{'temperature':0.}}),
                  dict(training={'loss':{'exclude_visited':'yes'}})]:
        with pytest.raises((ValueError,TypeError)):
            load_config(config=cfg,overrides=patch)
