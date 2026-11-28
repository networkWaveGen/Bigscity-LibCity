"""Physical CPT periods and causal horizons without hourly rounding."""
import numpy as np
import pytest

from libcity.data.cpt_offsets import cpt_offsets
from libcity.testing.fixtures import periodic_fixture
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


@pytest.mark.parametrize('interval,horizon',[(60,12),(300,12),(600,6),(1800,2),(3600,1)])
def test_existing_hour_divisor_protocol_offsets_and_layout_remain_identical(interval,horizon):
    sampling = dict(trend=2,period=1,closeness=2,interval_period=2,interval_trend=1)
    points = 3600//interval
    expected = np.concatenate([np.arange(-index*points*unit,-index*points*unit+horizon)
        for count,unit in ((2,168),(1,48),(2,1)) for index in range(1,count+1)])
    actual = cpt_offsets(sampling,interval,horizon)
    np.testing.assert_array_equal(actual,expected)
    assert actual.dtype==np.int64 and actual.max()<0
    assert int(-actual.min())==int(-expected.min())


@pytest.mark.parametrize('sampling,interval,horizon,shift',[
    (dict(trend=1),420,4,1440),
    (dict(period=1),7200,4,12),
    (dict(period=1,interval_period=7),420,4,1440),
    (dict(trend=1),86400,3,7),
    (dict(period=1,trend=1),7200,12,12)])
def test_only_active_physical_shifts_constrain_frequency_and_complete_history(sampling,interval,horizon,shift):
    actual = cpt_offsets(sampling,interval,horizon)
    assert actual.max()<0
    # The final active family is the closest physical shift in these examples.
    np.testing.assert_array_equal(actual[-horizon:],np.arange(-shift,-shift+horizon))
    if sampling.get('trend'):
        assert -actual[0]*interval==7*86400
    if sampling.get('period'):
        assert -actual[-horizon]*interval==86400*sampling.get('interval_period',1)


@pytest.mark.parametrize('sampling,interval',[
    (dict(closeness=1),420),(dict(period=1),420),(dict(closeness=1),7200),
    (dict(period=1,trend=1),420)])
def test_unrepresentable_active_periods_fail_instead_of_silently_drifting(sampling,interval):
    with pytest.raises(ValueError,match='exactly representable'):
        cpt_offsets(sampling,interval,1)


@pytest.mark.parametrize('sampling,interval,horizon',[
    (dict(closeness=1),300,13),(dict(period=1),7200,13),(dict(trend=1),86400,8),
    (dict(period=1,trend=1),7200,13)])
def test_each_complete_input_window_ends_before_the_target(sampling,interval,horizon):
    with pytest.raises(ValueError,match='inputs would reach the target'):
        cpt_offsets(sampling,interval,horizon)


@pytest.mark.parametrize('interval',(0,-1,True,300.))
def test_invalid_source_interval_is_rejected(interval):
    with pytest.raises(ValueError,match='time_intervals'):
        cpt_offsets(dict(closeness=1),interval,1)


def test_two_hour_daily_only_bundle_has_exact_day_alignment_and_disjoint_targets(tmp_path):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=120,nodes=3,features=2,outputs=2,interval_minutes=120)
    cfg = dict(model=dict(name='DGCN',params=dict(c_out=4,K=2,dropout=0)),
        data=dict(dataset='Tiny',data_dir=str(tmp_path/'raw'),cache_dir=str(tmp_path/'cache'),
            input_window=4,output_window=4,pad_training=False,sampling=dict(name='cpt',closeness=0,period=1,trend=0)),
        runtime=dict(device='cpu',output_dir=str(tmp_path/'runs'),exp_id='day'))
    context = make_context(cfg)
    try:
        bundle = context.bundle
        train,val,test = (getattr(bundle,name).dataset for name in ('train','val','test'))
        np.testing.assert_array_equal(train.offsets,[-12,-11,-10,-9])
        start = int(val.target_starts[0])
        observed = train.clock[start+train.offsets]
        targets = train.clock[start+np.arange(4)]
        np.testing.assert_array_equal(targets-observed,np.full(4,np.timedelta64(1,'D')))
        assert train.target_starts.max()+3<val.target_starts.min()
        assert val.target_starts.max()+3<test.target_starts.min()
        assert bundle.split_summary['history_span_points']==12
        assert (len(train),len(val),len(test))==(57,21,21)
        np.testing.assert_allclose(bundle.scaler.mean,train.targets[:72].mean(),rtol=1e-6)
    finally:
        context.tracker.close()


def test_weekly_seven_minute_bundle_records_corrected_sampling_identity(tmp_path):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=3200,nodes=1,features=1,outputs=1,interval_minutes=7)
    cfg = dict(model=dict(name='DGCN',params=dict(c_out=4,K=2)),
        data=dict(dataset='Tiny',data_dir=str(tmp_path/'raw'),cache_dir=str(tmp_path/'cache'),input_window=4,
            output_window=4,pad_training=False,sampling=dict(name='cpt',closeness=0,period=0,trend=1)),
        runtime=dict(device='cpu',output_dir=str(tmp_path/'runs'),exp_id='week'))
    context = make_context(cfg)
    try:
        dataset = context.bundle.test.dataset
        start = int(dataset.target_starts[0])
        np.testing.assert_array_equal(dataset.clock[start+np.arange(4)]-dataset.clock[start+dataset.offsets],
            np.full(4,np.timedelta64(7,'D')))
        # max_position is already part of safe resume metadata. Correcting an
        # accepted but inexact old period must not keep the old identity.
        assert context.bundle.metadata.input_layout['max_position']==1440
        assert context.bundle.metadata.input_layout['max_position']!=(3600//420)*168
    finally:
        context.tracker.close()
