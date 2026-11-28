"""DSAN axes, local patches, chained attention and raw masked-RMSE lifecycle."""
import importlib
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.config.runtime import load_config,merge_config
from libcity.data.bundle import DatasetMetadata
from libcity.data.grid_traffic import prepare_grid
from libcity.data.sources import atomic_sources
from libcity.model.traffic_flow_prediction.runtime_dsan import DSAN,attention,grid_patches,spatial_encoding,decoder_seed
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import grid_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(rows=2,columns=3,features=3,outputs=2,external=2):
    nodes = rows*columns
    return DatasetMetadata('tiny',nodes,features,outputs,input_layout=dict(kind='grid_window',grid_shape=[rows,columns],
        grid_order=list(range(nodes)),node_positions=list(range(nodes)),historical_external_dim=external,
        history_mask='finite_nonzero_raw_observations'))


def configuration(root,name):
    return dict(model=dict(name='DSAN',params=dict(L=2,d=8,n_h=2,B=1,l_d=1,r_d=.2)),
        data=dict(dataset='GridTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=3,pad_training=False),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,
            scheduler=dict(name='plateau',interval='metric',monitor='val_loss',factor=.5,patience=0)),
        evaluation=dict(batch_size=4),runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def reference_attention(Q,K,V,M,n_h):
    logits = Q@K.transpose(-1,-2)/math.sqrt(K.shape[-1])
    if M is not None:
        mask = M.bool().unsqueeze(2)
        logits = torch.where(mask.all(-1,keepdim=True),0.,logits.masked_fill(mask,float('-inf')))
        weights = logits.softmax(-1).masked_fill(mask,0.)
    else:
        weights = logits.softmax(-1)
    return weights@V


def loop_reference(model,current,source,valid,external,rows,columns,radius,horizon):
    """Legacy Torch blocks with independent per-sample/per-region alignment.

    Correct the head scale, chain every layer, convolve each actual local
    patch, allow the causal diagonal, and return physical row-major nodes.
    """
    results = []
    dae,sad = model.dsan.dae,model.dsan.sad
    steps,nodes,outputs = source.shape[1:]
    size = 2*radius+1
    for batch in range(len(source)):
        values = source[batch:batch+1]
        observed = valid[batch:batch+1]
        ex = external[batch:batch+1]
        grid = values.reshape(1,steps,rows,columns,outputs)
        global_values = dae.convs_g(grid).reshape(1,steps,nodes,-1)*math.sqrt(current.dsan.final_layer.in_features)
        global_values = global_values+dae.ex_encoder_g(ex)+current.global_positions
        global_mask = ~observed.any(-1).unsqueeze(2)
        for layer in dae.Enc_G:
            global_values = layer(global_values,global_mask)
        predicted = []
        image = F.pad(grid.permute(0,1,4,2,3),(radius,radius,radius,radius))
        known_image = F.pad(observed.any(-1).reshape(1,steps,rows,columns),(radius,radius,radius,radius))
        for node in range(nodes):
            row,column = divmod(node,columns)
            patch = image[...,row:row+size,column:column+size].permute(0,1,3,4,2)
            local = dae.convs_d(patch).reshape(1,steps,size*size,-1)*math.sqrt(current.dsan.final_layer.in_features)
            local = local+dae.ex_encoder_d(ex[:,:,node]).unsqueeze(2)+current.local_positions[:,0]
            local_mask = ~known_image[...,row:row+size,column:column+size].flatten(-2).unsqueeze(2)
            for layer in dae.Enc_D:
                local = layer(local,global_values,local_mask,global_mask)
            seeds = values[:,-horizon:,node]
            seed_ex = ex[:,-horizon:,node]
            encoded = sad.li_conv(seeds)*math.sqrt(current.dsan.final_layer.in_features)+sad.ex_encoder(seed_ex)+current.dsan.sad.pos_enc
            spatial = encoded.unsqueeze(2).expand(-1,-1,horizon,-1)
            memory = sad.linear(local.permute(0,2,3,1)).permute(0,3,1,2)
            seed_mask = ~observed[:,-horizon:,node].any(-1)
            blocked = seed_mask[:,None,None,:]|torch.ones(horizon,horizon,dtype=torch.bool).triu(1)
            for layer in sad.dec_s:
                spatial = layer(spatial,memory,blocked,None)
            temporal = encoded.unsqueeze(1)
            for layer in sad.dec_t:
                temporal = layer(temporal,spatial.permute(0,2,1,3),blocked,None)
            predicted.append(model.dsan.final_layer(temporal[:,0]).tanh())
        results.append(torch.stack(predicted,2))
    return torch.cat(results)


@pytest.mark.parametrize('rows,columns,outputs,features,ext,layers,width,heads,convs,radius,steps,horizon',[
    (2,2,2,3,2,3,64,8,3,1,3,2),(1,3,1,2,1,2,8,2,2,1,4,3),
    (2,1,3,3,2,1,12,3,1,2,2,2),(1,1,1,1,1,2,4,1,1,0,3,1)])
def test_corrected_legacy_blocks_match_full_per_sample_per_region_reference_and_all_gradients(
        monkeypatch,rows,columns,outputs,features,ext,layers,width,heads,convs,radius,steps,horizon):
    legacy = importlib.import_module('libcity.model.traffic_flow_prediction.DSAN')
    monkeypatch.setattr(legacy,'cal_attention',reference_attention)
    torch.manual_seed(17)
    meta = metadata(rows,columns,features,outputs,ext)
    old = legacy.DSAN(dict(input_window=steps,output_window=horizon,L=layers,d=width,n_h=heads,B=convs,l_d=radius,r_d=0.),
        dict(len_row=rows,len_column=columns,num_nodes=rows*columns,feature_dim=features+ext,
            output_dim=outputs,ext_dim=features-outputs+ext,scaler=NoneScaler()))
    for layer in old.dsan.dae.Enc_G:
        layer.ffn = legacy.two_layer_ffn(width,4*width,width)
    old.dsan.sad.li_conv.linear = torch.nn.Linear(outputs,width)
    current = DSAN(metadata=meta,input_window=steps,output_window=horizon,L=layers,d=width,n_h=heads,B=convs,l_d=radius,r_d=0.)
    loaded = current.load_state_dict(old.state_dict(),strict=False)
    assert not loaded.unexpected_keys and all('positions' in name or 'grid_order' in name or 'pos_enc' in name or 'causal_mask' in name for name in loaded.missing_keys)
    old,current = old.double().eval(),current.double().eval()
    source = torch.randn(2,steps,rows*columns,features,dtype=torch.float64,requires_grad=True)
    context = torch.randn(2,steps,ext,dtype=torch.float64,requires_grad=True)
    reference = source.detach().clone().requires_grad_()
    reference_ex = context.detach().clone().requires_grad_()
    valid = torch.rand_like(source)>.25
    valid[0,0] = False
    actual = current(dict(X=source,history_mask=valid,history_features=context))
    all_ex = torch.cat((reference[...,outputs:],reference_ex[:,:,None].expand(-1,-1,rows*columns,-1)),-1)
    expected = loop_reference(old,current,reference[...,:outputs],valid[...,:outputs],all_ex,rows,columns,radius,horizon)
    torch.testing.assert_close(actual,expected,rtol=2e-9,atol=2e-10)
    actual.square().mean().backward();expected.square().mean().backward()
    torch.testing.assert_close(source.grad,reference.grad,rtol=2e-7,atol=2e-10)
    torch.testing.assert_close(context.grad,reference_ex.grad,rtol=2e-7,atol=2e-10)
    previous = dict(old.named_parameters())
    for name,value in current.named_parameters():
        assert value.grad is not None and previous[name].grad is not None,name
        torch.testing.assert_close(value.grad,previous[name].grad,rtol=3e-7,atol=3e-10)


@pytest.mark.parametrize('kind',['none','partial','empty'])
def test_attention_scale_mask_and_finite_all_blocked_gradients_match_torch_sdpa(kind):
    torch.manual_seed(17)
    query = torch.randn(2,3,2,3,dtype=torch.float64,requires_grad=True)
    key = torch.randn(2,3,4,3,dtype=torch.float64,requires_grad=True)
    value = torch.randn_like(key,requires_grad=True)
    blocked = None if kind=='none' else torch.ones(2,1,2,4,dtype=torch.bool)
    if kind=='partial':
        blocked[0,0,1,:2] = False
        blocked[1,0,:,:3] = False
    actual = attention(query,key,value,blocked)
    expected = F.scaled_dot_product_attention(query,key,value,attn_mask=None if blocked is None else ~blocked)
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    first = torch.autograd.grad(actual.square().sum(),(query,key,value),retain_graph=True)
    second = torch.autograd.grad(expected.square().sum(),(query,key,value))
    for a,b in zip(first,second,strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('width',[1,5,64])
def test_registered_spatial_encoding_matches_independent_numpy_angles(width):
    coordinates = torch.tensor([[-2.,3.],[0.,0.],[1.,-1.]])
    rates = 10000**(-(2*(np.arange(width)//2))/width)
    expected = np.stack([np.where(np.arange(width)%2==0,np.sin(row*rates),np.cos(column*rates)) for row,column in coordinates.numpy()])
    np.testing.assert_allclose(spatial_encoding(coordinates,width).numpy(),expected,rtol=1e-6,atol=5e-8)


@pytest.mark.parametrize('radius',[0,1,2])
def test_vectorized_local_windows_and_boolean_masks_match_explicit_padded_grid(radius):
    values = torch.arange(2*3*2*3*2,dtype=torch.float64).reshape(2,3,2,3,2).requires_grad_()
    side = radius*2+1
    padded = F.pad(values.permute(0,1,4,2,3),(radius,radius,radius,radius))
    expected = torch.stack([padded[...,row:row+side,column:column+side].permute(0,1,3,4,2)
        for row in range(2) for column in range(3)],2)
    actual = grid_patches(values,radius)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    first = torch.autograd.grad(actual.square().sum(),values,retain_graph=True)[0]
    second = torch.autograd.grad(expected.square().sum(),values)[0]
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert torch.equal(grid_patches(values.detach()!=0,radius),expected!=0)


@pytest.mark.parametrize('empty,external,horizon',[(False,2,2),(True,0,5)])
def test_batch_independence_no_labels_future_inputs_or_rng_and_longer_horizon(empty,external,horizon):
    model = DSAN(metadata=metadata(1,2,2,2,external),input_window=3,output_window=horizon,L=2,d=6,n_h=2,B=1,l_d=1,r_d=.2).double().eval()
    source = torch.randn(3,3,2,2,dtype=torch.float64,requires_grad=True)
    mask = torch.zeros_like(source,dtype=torch.bool) if empty else torch.ones_like(source,dtype=torch.bool)
    inputs = dict(X=source,history_mask=mask,history_features=torch.randn(3,3,external,dtype=torch.float64))
    state = torch.get_rng_state().clone()
    actual = model(inputs)
    separate = torch.cat([model({key:value[index:index+1] for key,value in inputs.items()}) for index in range(3)])
    torch.testing.assert_close(actual,separate,rtol=1e-11,atol=1e-12)
    gradient = torch.autograd.grad(actual[0].sum(),source)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient[1:])==0
    dirty = dict(inputs,y=torch.full_like(actual,float('nan')),future_features=torch.full((3,horizon,7),float('nan')))
    torch.testing.assert_close(model(dirty),actual,rtol=0,atol=0)
    assert torch.equal(state,torch.get_rng_state())
    assert not any(hasattr(model,name) for name in ('_scaler','logger','device'))
    if empty:
        assert model.dsan.dae.ex_encoder_d is None
        torch.testing.assert_close(decoder_seed(source,horizon)[:,-1],source[:,-1])


def test_contiguous_grid_raw_masks_train_statistics_future_isolation_and_source_manifest(tmp_path):
    folder = grid_fixture(tmp_path/'raw',steps=60)
    (folder/'GridTiny.rel').write_text('unused bad relation',encoding='utf-8')
    cfg = load_config(config=configuration(tmp_path,'data'))
    prepare = lambda:prepare_grid(cfg['data'],cfg['training'],cfg['evaluation'],17)
    bundle = prepare()
    assert np.array_equal(bundle.train.dataset.offsets,[-4,-3,-2,-1])
    raw = bundle.train.dataset.arrays['history']
    assert bundle.scaler.mean==pytest.approx(raw[:36,:,:2].mean())
    sample = bundle.train.dataset[0]
    assert sample['history_mask'].dtype==torch.bool
    np.testing.assert_equal(sample['history_mask'],raw[:4]!=0)
    assert torch.any(~sample['history_mask']&(sample['X']!=0))
    sample = bundle.val.dataset[0]
    frame = pd.read_csv(folder/'GridTiny.grid');times=sorted(frame['time'].unique())
    frame.loc[frame['time'].isin(times[36:]),['value_0','value_1']] += 1000
    frame.to_csv(folder/'GridTiny.grid',index=False)
    external = pd.read_csv(folder/'GridTiny.ext')
    external.loc[external['time'].isin(times[36:]),'temperature'] += 500
    external.to_csv(folder/'GridTiny.ext',index=False)
    changed = prepare()
    assert vars(bundle.scaler)==vars(changed.scaler)
    for key in ('X','history_mask','history_features','future_features'):
        torch.testing.assert_close(sample[key],changed.val.dataset[0][key],rtol=0,atol=0)
    records = atomic_sources(cfg['data'])
    assert any(row['name']=='GridTiny.grid' for row in records)
    assert not any(row['name']=='GridTiny.dyna' and row['size'] is None for row in records)
    sets = [{int(start+step) for start in loader.dataset.target_starts for step in range(3)} for loader in (bundle.train,bundle.val,bundle.test)]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]


def test_node_identity_permutation_preserves_rectangular_grid_and_output_channels():
    torch.manual_seed(17)
    meta = metadata()
    positions = np.array([4,0,5,2,1,3])
    shuffled = replace(meta,input_layout=dict(meta.input_layout,node_positions=positions,grid_order=np.argsort(positions)))
    base = DSAN(metadata=meta,input_window=3,output_window=2,L=2,d=8,n_h=2,B=1,l_d=1,r_d=0.).double().eval()
    other = DSAN(metadata=shuffled,input_window=3,output_window=2,L=2,d=8,n_h=2,B=1,l_d=1,r_d=0.).double().eval()
    params = dict(base.named_parameters())
    with torch.no_grad():
        for name,value in other.named_parameters(): value.copy_(params[name])
    source = torch.randn(2,3,6,3,dtype=torch.float64)
    mask = torch.rand_like(source)>.2
    external = torch.randn(2,3,2,dtype=torch.float64)
    expected = base(dict(X=source,history_mask=mask,history_features=external))[:,:,positions]
    actual = other(dict(X=source[:,:,positions],history_mask=mask[:,:,positions],history_features=external))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_nan_history_is_imputed_without_changing_raw_targets_or_nonzero_mask(tmp_path):
    folder = grid_fixture(tmp_path/'raw',steps=60)
    path = folder/'GridTiny.grid';frame=pd.read_csv(path)
    clock = sorted(frame['time'].unique())
    selection=(frame['time']==clock[6])&(frame['row_id']==0)&(frame['column_id']==0)
    frame.loc[selection,'value_0']=np.nan;frame.to_csv(path,index=False)
    context=make_context(configuration(tmp_path,'missing'))
    try:
        dataset=context.bundle.train.dataset
        sample=dataset[list(dataset.target_starts).index(7)]
        node=list(context.bundle.metadata.input_layout['node_positions']).index(0)
        assert not sample['history_mask'][-1,node,0] and torch.isfinite(sample['X']).all()
        assert torch.isnan(dataset[list(dataset.target_starts).index(6)]['y_raw'][0,node,0])
    finally: context.tracker.close()


def test_non_cpt_window_accepts_exact_non_day_dividing_clock_intervals(tmp_path):
    folder=grid_fixture(tmp_path/'raw',steps=60)
    for suffix in ('.grid','.ext'):
        path=folder/('GridTiny'+suffix);frame=pd.read_csv(path)
        times=sorted(frame['time'].unique())
        replacements={value:(pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(seconds=index*777)).strftime('%Y-%m-%dT%H:%M:%SZ') for index,value in enumerate(times)}
        frame['time']=frame['time'].map(replacements);frame.to_csv(path,index=False)
    path=folder/'config.json';raw=json.loads(path.read_text());raw['time_intervals']=777;path.write_text(json.dumps(raw))
    context=make_context(configuration(tmp_path,'clock'))
    try:
        clock=context.bundle.train.dataset.arrays['clock']
        assert np.all(np.diff(clock)==np.timedelta64(777,'s'))
        assert context.bundle.metadata.info['time_intervals']==777
    finally: context.tracker.close()


def test_grid_cache_hit_reads_no_observation_or_external_rows(tmp_path,monkeypatch):
    grid_fixture(tmp_path/'raw',steps=60)
    cfg=load_config(config=configuration(tmp_path,'cached'))
    prepare=lambda:prepare_grid(cfg['data'],cfg['training'],cfg['evaluation'],17)
    prepare()
    original=pd.read_csv
    def guarded(path,*args,**kwargs):
        if Path(path).suffix in ('.grid','.ext') and kwargs.get('nrows')!=0:
            pytest.fail('Cache hit re-read complete observations/covariates')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_csv',guarded)
    assert prepare().split_summary['cache']['status']=='hit'


@pytest.mark.parametrize('damage',['dtype','known','clock','intervals'])
def test_raw_grid_cache_reuses_new_window_and_rebuilds_invalid_arrays(tmp_path,damage):
    grid_fixture(tmp_path/'raw',steps=60)
    cfg = load_config(config=configuration(tmp_path,'cache'))
    prepare = lambda:prepare_grid(cfg['data'],cfg['training'],cfg['evaluation'],17)
    first = prepare();path=first.split_summary['cache']['path']
    cfg['data'].update(input_window=2,output_window=1,scaler=dict(name='none'))
    assert prepare().split_summary['cache']['status']=='hit'
    with np.load(path,allow_pickle=False) as archive:
        arrays = {name:archive[name] for name in archive.files}
    if damage=='dtype': arrays['history']=np.full(arrays['history'].shape,'bad')
    elif damage=='known': arrays['known'][0,0]=np.inf
    elif damage=='clock': arrays['clock']=arrays['clock'][0]
    else: arrays['intervals']=np.array([[0,59]])
    np.savez_compressed(path,**arrays)
    assert prepare().split_summary['cache']['status']=='miss'


def equal(left,right):
    if isinstance(left,torch.Tensor): torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True): equal(a,b)
    else: assert left==right


@pytest.mark.parametrize('scaler',['standard','minmax01'])
def test_real_grid_masked_rmse_dropout_all_layers_independent_evaluation_and_exact_resume(tmp_path,scaler):
    grid_fixture(tmp_path/'raw',steps=60)
    def config(name):
        result=configuration(tmp_path,name);result['data']['scaler']=dict(name=scaler);return result
    trained = run_experiment('train',config=config('full'))
    evaluated = run_experiment('evaluate',config=config('eval'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        prediction,target=archive['prediction'],archive['truth'];valid=np.isfinite(target)&(target!=0)
        assert prediction.shape[1:]==(3,6,2)
        assert evaluated.metrics['loss']==pytest.approx(np.sqrt(np.square(prediction[valid]-target[valid]).mean()),rel=2e-6)
    partial=make_context(config('partial'))
    before=partial.model.dsan.dae.Enc_D[0].msa2.wq.weight.detach().clone()
    from libcity.executor.train import train_loop
    try:
        train_loop(partial,until_epoch=1)
        assert not torch.equal(before,partial.model.dsan.dae.Enc_D[0].msa2.wq.weight)
    finally: partial.tracker.close()
    resumed=run_experiment('train',config=config('resume'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual=[torch.load(result.artifacts['last'],weights_only=True) for result in (trained,resumed)]
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'): equal(expected[key],actual[key])


@pytest.mark.parametrize('singleton',[False,True])
def test_actual_cpu_amp_tail_single_grid_and_horizon_exceeding_history(tmp_path,singleton):
    grid_fixture(tmp_path/'raw',steps=50,rows=1 if singleton else 2,columns=1 if singleton else 3,features=2,outputs=2)
    cfg=configuration(tmp_path,'amp');cfg['training'].update(epochs=1,amp=True)
    if singleton: cfg['data'].update(input_window=1,features=dict(load_external=False,time_in_day=False,day_in_week=False))
    result=run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


def test_spawn_worker_transports_grid_raw_masks(tmp_path):
    grid_fixture(tmp_path/'raw',steps=60)
    cfg=configuration(tmp_path,'spawn');cfg['training']['num_workers']=1
    context=make_context(cfg)
    try:
        batch=next(iter(context.bundle.train))
        assert batch['X'].shape==(5,4,6,3) and batch['history_mask'].dtype==torch.bool
    finally: context.tracker.close()


@pytest.mark.parametrize('change',[
    dict(model=dict(params=dict(L=0))),dict(model=dict(params=dict(d=7))),dict(model=dict(params=dict(n_h=0))),
    dict(model=dict(params=dict(B=0))),dict(model=dict(params=dict(l_d=-1))),dict(model=dict(params=dict(r_d=1.))),
    dict(data=dict(sampling=dict(name='window'))),dict(data=dict(graph=dict(name='binary')))])
def test_invalid_dsan_configuration_is_rejected_at_startup(change):
    with pytest.raises(ValueError): load_config(config=merge_config(dict(model=dict(name='DSAN'),data=dict(dataset='missing')),change))
