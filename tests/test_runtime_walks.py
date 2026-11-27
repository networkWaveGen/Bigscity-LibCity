"""Faithful sparse walks, real Word2Vec fitting and honest external capabilities."""
import builtins
from collections import Counter
import json
from itertools import islice
import random
import subprocess
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.data.walk_graph import WalkCorpus,alias_table
from libcity.executor.walk_training import stable_token_hash,WalkLines
from libcity.model.road_representation.Node2Vec import Graph as ReferenceNode2Vec
from libcity.model.road_representation.DeepWalk import from_numpy,build_deepwalk_corpus
from libcity.model.runtime_registry import model_spec,resolve
from libcity.pipeline.experiment import run_experiment
from libcity.pipeline.runtime_tuning import run_search


def fixture(root):
    folder = root/'RoadWalk';folder.mkdir(parents=True)
    (folder/'config.json').write_text(json.dumps(dict(geo_file='RoadWalk',rel_file='RoadWalk',weight_col='affinity')),encoding='utf-8')
    pd.DataFrame(dict(geo_id=[30,10,70,20,40],type='LineString')).to_csv(folder/'RoadWalk.geo',index=False)
    edges = [(30,10,1.),(30,70,3.),(10,30,2.),(10,20,4.),(70,10,1.),(70,20,2.),(20,30,1.)]
    pd.DataFrame(dict(origin_id=[a for a,b,w in edges],destination_id=[b for a,b,w in edges],affinity=[w for a,b,w in edges])).to_csv(folder/'RoadWalk.rel',index=False)
    return folder


def configuration(root,name,model='Node2Vec'):
    return dict(data=dict(dataset='RoadWalk',data_dir=str(root/'raw'),cache=False),
        model=dict(name=model,params=dict(output_dim=4,num_walks=3,walk_length=8)),
        training=dict(epochs=2,window_size=3,subsample=0.),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


def prepare(cfg):
    cfg = load_config(config=cfg);spec = model_spec(cfg['model']['name'])
    bundle = resolve(spec.prepare_data)(cfg['data'],cfg['training'],cfg['evaluation'],cfg['runtime']['seed'])
    model = resolve(spec.factory)(metadata=bundle.metadata,**cfg['model']['params'])
    return cfg,bundle,model


@pytest.mark.parametrize('directed',[False,True])
def test_node2vec_walks_match_original_alias_sampling_without_global_rng_mutation(tmp_path,directed):
    fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'walks');cfg['data']['graph'] = dict(name='weighted',directed=directed)
    cfg['model']['params'].update(p=2.,q=.5)
    cfg,bundle,model = prepare(cfg)
    info = bundle.metadata.info
    dense = np.zeros((5,5));dense[tuple(info['edge_index'])] = info['edge_weights']
    graph = nx.from_numpy_array(dense,create_using=nx.DiGraph() if directed else nx.Graph())
    reference = ReferenceNode2Vec(graph,directed,2.,.5);reference.preprocess_transition_probs()
    python_state,numpy_state = random.getstate(),np.random.get_state()
    try:
        random.seed(17);np.random.seed(17)
        expected = [[str(node) for node in walk] for walk in reference.simulate_walks(3,8)]
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)
    corpus = WalkCorpus(bundle.metadata,algorithm='node2vec',num_walks=3,walk_length=8,seed=17,p=2.,q=.5,cache_budget=7)
    assert list(corpus)==expected and list(corpus)==expected and corpus.cache_size<=7
    assert random.getstate()==python_state
    current = np.random.get_state()
    assert current[0]==numpy_state[0] and current[2:]==numpy_state[2:]
    np.testing.assert_array_equal(current[1],numpy_state[1])
    assert Counter(int(walk[0]) for walk in expected)==Counter({node:3 for node in range(5)})
    assert bundle.metadata.adjacency is None and bundle.train is bundle.val is bundle.test is None


@pytest.mark.parametrize('restart',[0.,.4,1.])
def test_deepwalk_matches_corrected_zero_start_and_default_rng_is_not_shared(tmp_path,restart):
    fixture(tmp_path/'raw')
    _,bundle,_ = prepare(configuration(tmp_path,'deep','DeepWalk'))
    dense = np.zeros((5,5));dense[tuple(bundle.metadata.info['edge_index'])] = 1.
    reference = from_numpy(dense,directed=False)
    expected = build_deepwalk_corpus(reference,3,8,alpha=restart,rand=random.Random(17))
    corpus = WalkCorpus(bundle.metadata,algorithm='deepwalk',num_walks=3,walk_length=8,seed=17,restart=restart)
    assert list(corpus)==expected
    assert reference.random_walk(1,rand=random.Random(17),start=0)==['0']
    assert build_deepwalk_corpus(reference,2,5)==build_deepwalk_corpus(reference,2,5)
    assert Counter(int(walk[0]) for walk in expected)==Counter({node:3 for node in range(5)})


def test_transition_probabilities_alias_distribution_and_weighted_symmetry(tmp_path):
    folder = fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'probabilities');cfg['data']['graph'] = dict(name='weighted')
    _,bundle,_ = prepare(cfg)
    corpus = WalkCorpus(bundle.metadata,algorithm='node2vec',num_walks=1,walk_length=3,seed=17,p=2.,q=.5)
    # Current node 1 -> [0,3]: return weight 2/p vs adjacent weight 4.
    np.testing.assert_allclose(corpus.probabilities(0,1),[.2,.8])
    corpus.q = .25
    # For the reachable 2 -> 1 transition, candidate 3 has no edge back to 2.
    np.testing.assert_allclose(corpus.probabilities(2,1),[1/9,8/9])
    aliases,threshold = alias_table([.1,.2,.7]);rng = np.random.RandomState(17)
    values = []
    for _ in range(20000):
        index = int(rng.rand()*3);values.append(index if rng.rand()<threshold[index] else aliases[index])
    np.testing.assert_allclose(np.bincount(values,minlength=3)/20000,[.1,.2,.7],atol=.01)
    cfg['data']['graph']['directed'] = False
    _,symmetric,_ = prepare(cfg)
    edges = {tuple(edge):weight for edge,weight in zip(symmetric.metadata.info['edge_index'].T,symmetric.metadata.info['edge_weights'])}
    assert edges[0,1]==edges[1,0]==2.
    assert symmetric.metadata.info['node_ids']==(30,10,70,20,40)
    assert symmetric.metadata.info['isolated_nodes']==1
    with pytest.raises(ValueError): symmetric.metadata.info['edge_weights'][0] = 3
    relations = pd.read_csv(folder/'RoadWalk.rel').iloc[:1]
    relations.to_csv(folder/'RoadWalk.rel',index=False)
    _,directed,_ = prepare(configuration(tmp_path,'sinks'))
    assert directed.metadata.info['isolated_nodes']==3 and directed.metadata.info['dead_end_nodes']==4


@pytest.mark.parametrize('name',['Node2Vec','DeepWalk'])
def test_real_word2vec_reference_weights_safe_export_and_no_torch_optimizer(tmp_path,monkeypatch,name):
    gensim = pytest.importorskip('gensim')
    fixture(tmp_path/'raw');cfg = configuration(tmp_path,'fit',name)
    def reject(*args,**kwargs): raise AssertionError('External Word2Vec must not create a PyTorch optimizer or writer')
    monkeypatch.setattr('libcity.executor.train.build_context',reject)
    monkeypatch.setattr('libcity.pipeline.experiment.build_context',reject)
    monkeypatch.setattr('libcity.executor.train.MetricTracker',reject)
    result = run_experiment('train',config=cfg)
    assert 'best_validation' not in result.metrics and not {'best','last'}&result.artifacts.keys()
    resolved,bundle,model = prepare(cfg)
    weights = torch.load(result.artifacts['weights'],weights_only=True)
    assert all(isinstance(value,torch.Tensor) for value in weights.values())
    model.load_state_dict(weights)
    assert not list(model.parameters())
    model.double()
    torch.testing.assert_close(model({'node_indices':torch.tensor([0,4])}),weights['vectors'][[0,4]].double(),rtol=0,atol=0)
    options = resolved['training']
    reference = gensim.models.Word2Vec(WalkLines(result.artifacts['walks']),vector_size=4,window=3,min_count=0,sg=1,
        hs=int(options['loss']['hierarchical_softmax']),negative=options['loss']['negative_samples'],workers=1,
        epochs=2,alpha=.025,min_alpha=.0001,sample=0.,shrink_windows=True,batch_words=10000,seed=17,
        hashfxn=stable_token_hash,compute_loss=True)
    expected = np.stack([reference.wv[str(index)] for index in range(5)])
    actual = np.load(result.artifacts['embeddings'],allow_pickle=False)
    np.testing.assert_array_equal(actual,expected)
    assert json.loads(result.artifacts['node_ids'].read_text())==[30,10,70,20,40]
    report = json.loads(result.artifacts['training_report'].read_text())
    assert report['validation'] is False and report['resume'] is False and report['workers']==1
    assert report['hierarchical_softmax']==(name=='DeepWalk') and report['negative_samples']==5
    assert report['walks']==15 and report['isolated_nodes']==1 and report['atomic_sources']
    records = [json.loads(line) for line in result.artifacts['training_metrics'].read_text().splitlines()]
    assert [record['epoch'] for record in records]==[1,2] and all(record['stage']=='word2vec' for record in records)
    evaluated = run_experiment('evaluate',config={**cfg,'runtime':{**cfg['runtime'],'exp_id':'evaluate'}},checkpoint=result.artifacts['weights'])
    assert evaluated.metrics==result.metrics
    cfg['evaluation'] = dict(save_predictions=False);cfg['runtime']['exp_id'] = 'evaluate'
    hidden = run_experiment('evaluate',config=cfg,checkpoint=result.artifacts['weights'])
    assert 'embeddings' not in hidden.artifacts and not evaluated.artifacts['embeddings'].exists() and not evaluated.artifacts['node_ids'].exists()
    assert not (tmp_path/'raw'/'RoadWalk'/'cache').exists()


@pytest.mark.parametrize('name',['Node2Vec','DeepWalk'])
def test_cross_process_determinism_and_evaluation_without_gensim(tmp_path,name):
    pytest.importorskip('gensim')
    fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'first',name)
    first = run_experiment('train',config=cfg)
    second_cfg = {**cfg,'runtime':{**cfg['runtime'],'exp_id':'second'}}
    script = """
import json,sys
from libcity.pipeline.experiment import run_experiment
result=run_experiment('train',config=json.loads(sys.argv[1]))
print(json.dumps({key:str(value) for key,value in result.artifacts.items()}))
"""
    process = subprocess.run([sys.executable,'-B','-c',script,json.dumps(second_cfg)],cwd=tmp_path,capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    second = json.loads(process.stdout)
    np.testing.assert_array_equal(np.load(first.artifacts['embeddings'],allow_pickle=False),np.load(second['embeddings'],allow_pickle=False))
    script = """
import builtins,json,sys
original=builtins.__import__
def blocked(name,*args,**kwargs):
    if name.split('.')[0]=='gensim': raise ImportError('gensim deliberately unavailable')
    return original(name,*args,**kwargs)
builtins.__import__=blocked
from libcity.pipeline.experiment import run_experiment
cfg=json.loads(sys.argv[1]);cfg['runtime']['exp_id']='no-gensim'
result=run_experiment('evaluate',config=cfg,checkpoint=sys.argv[2])
assert 'gensim' not in sys.modules
print(json.dumps(result.metrics))
"""
    process = subprocess.run([sys.executable,'-B','-c',script,json.dumps(cfg),str(first.artifacts['weights'])],cwd=tmp_path,capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    assert json.loads(process.stdout)==first.metrics


def test_capabilities_invalid_config_and_missing_training_dependency(tmp_path,monkeypatch):
    fixture(tmp_path/'raw');cfg = configuration(tmp_path,'invalid')
    with pytest.raises(ValueError,match='support --resume'): run_experiment('train',config=cfg,resume=tmp_path/'last.pt')
    with pytest.raises(ValueError,match='Validation-only'): run_experiment('train',config=cfg,evaluate_test=False)
    with pytest.raises(ValueError,match='Tuning requires'): run_search(config=cfg,space={'training.window_size':[2,3]})
    for overrides in [dict(training={'amp':True}),dict(training={'num_workers':2}),dict(runtime={'tensorboard':True}),
                      dict(training={'optimizer':{'name':'adam'}}),dict(evaluation={'metrics':['accuracy']}),dict(data={'split':{'train':.6,'val':.2}})]:
        with pytest.raises((ValueError,TypeError)): load_config(config=cfg,overrides=overrides)
    original = builtins.__import__
    def blocked(name,*args,**kwargs):
        if name.split('.')[0]=='gensim': raise ImportError('blocked')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    with pytest.raises(ImportError,match='embeddings'): run_experiment('train',config=cfg)


@pytest.mark.parametrize('changed',['node_ids','algorithm','graph','vectors'])
def test_safe_weights_reject_changed_node_mapping_algorithm_and_graph(tmp_path,changed):
    pytest.importorskip('gensim')
    fixture(tmp_path/'raw');cfg = configuration(tmp_path,'weights')
    result = run_experiment('train',config=cfg)
    _,_,model = prepare(cfg)
    weights = torch.load(result.artifacts['weights'],weights_only=True)
    if changed=='node_ids': weights['node_ids_utf8'][0] = 1
    elif changed=='algorithm': weights['algorithm_utf8'][0] = 1
    elif changed=='graph': weights['graph_weights'][0] += 1
    else: weights['vectors'][0,0] = float('nan')
    with pytest.raises(ValueError): model.load_state_dict(weights)
    assert not model.fitted.item()


def test_isolated_only_graph_is_explicit_and_sparse_without_full_matrix(tmp_path):
    pytest.importorskip('gensim')
    folder = fixture(tmp_path/'raw')
    pd.read_csv(folder/'RoadWalk.rel').iloc[:0].to_csv(folder/'RoadWalk.rel',index=False)
    cfg = configuration(tmp_path,'isolated')
    result = run_experiment('train',config=cfg)
    assert result.metrics['isolated_nodes']==5 and result.metrics['coverage']==1.
    report = json.loads(result.artifacts['training_report'].read_text())
    assert report['walk_tokens']==15 and report['word2vec_loss']==0.
    _,bundle,_ = prepare(cfg)
    assert bundle.metadata.info['edge_index'].shape==(2,0)
    np.testing.assert_array_equal(bundle.metadata.info['row_offsets'],np.zeros(6))


def test_large_sparse_graph_keeps_edge_storage_and_bounded_walk_cache(tmp_path):
    folder = fixture(tmp_path/'raw')
    count = 2000
    pd.DataFrame(dict(geo_id=np.arange(count)+10000)).to_csv(folder/'RoadWalk.geo',index=False)
    origins = np.repeat(np.arange(count)+10000,2)
    destinations = np.asarray([10000+(node+step)%count for node in range(count) for step in (1,2)])
    pd.DataFrame(dict(origin_id=origins,destination_id=destinations,affinity=1.)).to_csv(folder/'RoadWalk.rel',index=False)
    _,bundle,model = prepare(configuration(tmp_path,'sparse'))
    assert bundle.metadata.adjacency is None and bundle.metadata.info['edge_index'].shape==(2,4000)
    assert model.vectors.shape==(2000,4)
    corpus = WalkCorpus(bundle.metadata,algorithm='node2vec',num_walks=1,walk_length=20,seed=17,cache_budget=50)
    assert len(list(islice(corpus,20)))==20 and corpus.cache_size<=50
