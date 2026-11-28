"""A single observed training series for graph learning, independent of windows."""
from dataclasses import replace

import numpy as np

from libcity.data.timeline import boundaries
from libcity.data.traffic_bundle import prepare_traffic


def prepare_gts(data,training,evaluation,seed,logger=None):
    options = data['graph']
    plain = {**data,'graph':{key:value for key,value in options.items() if key not in ('history_channel','history_steps')}}
    bundle = prepare_traffic(plain,training,evaluation,seed,logger)
    metadata = bundle.metadata
    if metadata.input_layout['kind']!='window':
        raise ValueError('GTS requires continuous window sampling')
    channel,limit = options.get('history_channel',0),options.get('history_steps',0)
    if channel not in metadata.input_layout['observed_feature_indices']:
        raise ValueError('GTS history_channel must select an observed feature, not known calendar/future inputs')
    dataset = bundle.train.dataset
    end = boundaries(len(dataset.clock),data['split'],dataset.clock)[0]
    start = max(0,end-limit) if limit else 0
    interval = np.timedelta64(metadata.info.get('time_intervals',300),'s')
    if np.any(np.diff(dataset.clock[start:end])!=interval):
        raise ValueError('GTS graph history spans timestamp gaps; choose history_steps within a contiguous training segment')
    values = dataset.history[start:end,:,channel].copy()
    if channel<metadata.output_dim:
        values = bundle.scaler.transform(values)
    if not np.isfinite(values).all():
        raise ValueError('GTS graph history must contain finite training observations')
    values = values.astype(np.float32)
    layout = dict(source='raw_training_interval',channel=channel,start_index=start,end_index_exclusive=end,
        first_timestamp=str(dataset.clock[start]),last_timestamp=str(dataset.clock[end-1]),
        shape=list(values.shape),normalization='target_scaler' if channel<metadata.output_dim else 'unscaled_observation')
    bundle.metadata = replace(metadata,input_layout={**metadata.input_layout,'graph_history':layout},
        static_data={**metadata.static_data,'graph_history':values})
    bundle.split_summary['graph_history'] = layout
    return bundle
