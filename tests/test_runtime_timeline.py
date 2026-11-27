import json
import logging
import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.data.bundle import prepare_traffic
from libcity.executor.evaluate import ForecastLoss, model_inputs
from libcity.model.traffic_speed_prediction.runtime_rnn import GRU
from libcity.data.dataset.dataset_subclass.crann_dataset import CRANNDataset
from runtime_fixtures import atomic_fixture


def setup(tmp_path, **changes):
    folder=atomic_fixture(tmp_path/'raw',steps=100)
    cfg=load_config(config=dict(model={'name':'GRU','params':{'hidden_size':4}},
        data=dict(dataset='Tiny',data_dir=str(tmp_path/'raw'),cache_dir=str(tmp_path/'cache'),
                  input_window=4,output_window=3,pad_training=False,**changes),
        training=dict(batch_size=8,shuffle=False),runtime=dict(output_dir=str(tmp_path/'runs'))))
    return folder,cfg


def prepare(cfg):
    return prepare_traffic(cfg['data'],cfg['training'],cfg['evaluation'],17)


def test_target_windows_are_disjoint_and_history_can_precede_boundary(tmp_path):
    _,cfg=setup(tmp_path)
    bundle=prepare(cfg)
    starts=[loader.dataset.target_starts for loader in (bundle.train,bundle.val,bundle.test)]
    targets=[set((values[:,None]+np.arange(3)).ravel()) for values in starts]
    assert not targets[0]&targets[1] and not targets[1]&targets[2] and not targets[0]&targets[2]
    assert starts[1][0]==60 and starts[2][0]==80
    assert starts[1][0]+bundle.val.dataset.offsets.min()<60
    assert bundle.split_summary['discarded_boundary_windows']==4
    raw=pd.read_csv(tmp_path/'raw/Tiny/Tiny.dyna')
    expected=raw.groupby('entity_id',sort=False).head(60)['speed'].to_numpy()
    assert bundle.scaler.mean==pytest.approx(expected.mean())
    before=bundle.scaler.mean
    raw.loc[raw.groupby('entity_id',sort=False).cumcount()>=60,'speed']=1e8
    raw.to_csv(tmp_path/'raw/Tiny/Tiny.dyna',index=False)
    assert prepare(cfg).scaler.mean==before


def test_future_inputs_are_explicit_and_never_read_targets_or_future_observations(tmp_path):
    folder,cfg=setup(tmp_path,features=dict(time_in_day=True))
    frame=pd.read_csv(folder/'Tiny.dyna')
    frame['congestion']=np.arange(len(frame))
    frame.to_csv(folder/'Tiny.dyna',index=False)
    meta=json.loads((folder/'config.json').read_text())
    meta['data_col']=['speed','congestion']
    (folder/'config.json').write_text(json.dumps(meta))
    bundle=prepare(cfg)
    batch=next(iter(bundle.val))
    assert batch['y'].shape[-1]==1 and batch['future_features'].shape[-1]==1
    assert bundle.metadata.input_layout['future_feature_indices']==(2,)
    model=GRU(metadata=bundle.metadata,input_window=4,output_window=3,hidden_size=4).eval()
    without_labels=model_inputs({k:v for k,v in batch.items() if k not in ('y','y_raw')},1)
    prediction=model(without_labels)
    poisoned={**batch,'y':torch.full_like(batch['y'],float('nan')),'y_raw':torch.full_like(batch['y_raw'],1e9)}
    torch.testing.assert_close(prediction,model(model_inputs(poisoned,1)))
    assert set(without_labels)=={'X','future_features'}
    original=prediction[:1].detach()
    positions=frame.groupby('entity_id',sort=False).cumcount()
    frame.loc[positions>=60,'congestion']=1e9
    frame.to_csv(folder/'Tiny.dyna',index=False)
    changed=next(iter(prepare(cfg).val))
    # The first validation history is unchanged; unavailable future congestion
    # cannot alter the prediction even when its actual future values change.
    torch.testing.assert_close(original,model(model_inputs(changed,1))[:1])


def test_known_external_columns_require_explicit_declaration(tmp_path):
    folder,cfg=setup(tmp_path,features=dict(load_external=True,known_future_columns=['holiday']))
    times=pd.read_csv(folder/'Tiny.dyna')['time'].drop_duplicates().to_numpy()
    pd.DataFrame(dict(ext_id=range(100),time=times,weather=np.arange(100),holiday=np.arange(100)%2)).to_csv(folder/'Tiny.ext',index=False)
    bundle=prepare(cfg)
    batch=next(iter(bundle.val))
    assert batch['X'].shape[-1]==3 and batch['future_features'].shape[-1]==1
    assert bundle.metadata.input_layout['future_feature_indices']==(2,)
    cfg['data']['features']['known_future_columns']=['speed']
    with pytest.raises(ValueError,match='observations/targets'):
        prepare(cfg)


def test_raw_zero_mask_survives_normalization_rounding(tmp_path):
    _,cfg=setup(tmp_path)
    bundle=prepare(cfg)
    batch=next(iter(bundle.train))
    loss=ForecastLoss(dict(name='masked_mae',mask_value=0.),bundle.scaler,1)
    # The serialized raw target determines validity even if scaled values drift.
    perturbed=batch['y']+1e-5
    count=loss(perturbed,perturbed,batch['y_raw']).weight
    assert count==int((batch['y_raw']!=0).sum())<batch['y_raw'].numel()
    _,actual=loss.raw(perturbed,perturbed,batch['y_raw'])
    torch.testing.assert_close(actual,batch['y_raw'],rtol=0,atol=0)


def test_explicit_timestamp_boundaries_and_invalid_short_splits(tmp_path):
    _,cfg=setup(tmp_path,split=dict(train=.7,val=.1,train_end='2024-01-01T05:00:00',val_end='2024-01-01T06:00:00'))
    bundle=prepare(cfg)
    assert bundle.val.dataset.target_starts[0]==60
    assert bundle.test.dataset.target_starts[0]==72
    cfg['data']['split']['val_end']='2024-01-01T05:05:00'
    with pytest.raises(ValueError,match='val has no complete'):
        prepare(cfg)


def test_crann_all_four_fields_follow_disjoint_target_intervals():
    dataset=object.__new__(CRANNDataset)
    dataset.train_rate,dataset.eval_rate=.6,.2
    dataset.n_timesteps,dataset.output_window=6,3
    dataset.target_starts=np.arange(6,58)
    dataset.cache_dataset=False
    dataset._logger=logging.getLogger(__name__)
    fields=[np.arange(52)[:,None]+100*index for index in range(4)]
    actual=dataset._split_train_val_test(*fields)
    intervals=[set((starts[:,None]+np.arange(3)).ravel()) for starts in dataset.split_target_starts]
    assert not intervals[0]&intervals[1] and not intervals[1]&intervals[2]
    for part in range(3):
        for feature in range(4):
            np.testing.assert_array_equal(actual[part*4+feature]-100*feature,actual[part*4])
