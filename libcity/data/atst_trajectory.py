"""Label-free full-vocabulary candidate geometry and training-only popularity."""
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from libcity.data.geodesic import distances_from
from libcity.data.interval_trajectory import observed_prefixes
from libcity.data.task_bundles import collate_sequences,prepare_personalized_trajectory


class CandidateCollator:
    def __init__(self,coordinates):
        self.coordinates = np.asarray(coordinates,dtype=np.float64).copy()
        self.coordinates.setflags(write=False)
        # A bounded row cache avoids a dense vocabulary-squared distance matrix.
        self.cache = OrderedDict()

    def __call__(self,rows):
        batch = collate_sequences(rows)
        last = batch['current_loc'].gather(1,(batch['lengths']-1)[:,None]).flatten()-1
        values = []
        for identity in last.tolist():
            if identity not in self.cache:
                self.cache[identity] = torch.from_numpy(distances_from(self.coordinates[identity],self.coordinates).astype(np.float32))
                if len(self.cache)>16:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(identity)
            values.append(self.cache[identity])
        # Stack owns its memory: model/caller writes cannot change cached geometry.
        batch['candidate_dis'] = torch.stack(values)
        return batch


def prepare_atst(data,training,evaluation,seed,logger=None):
    result = prepare_personalized_trajectory(data,training,evaluation,seed,logger)
    prefixes,coordinates,groups = observed_prefixes(data,result.metadata.num_nodes,
        first_time=100.,first_distance=1.,model_name='ATSTLSTM')
    for loader in (result.train,result.val,result.test):
        for row in loader.dataset.rows:
            row.pop('current_tim')
            row['tim_interval'],row['dis'] = prefixes[int(row['sample_id'])]
        loader.collate_fn = CandidateCollator(coordinates)
    vocabulary = {identity:index for index,identity in enumerate(result.metadata.info['location_ids'])}
    weights = np.zeros(result.metadata.num_nodes,dtype=np.float64)
    end = pd.Timestamp(result.split_summary['train_end'])
    for group in groups:
        if len(group)<data['min_session_length']:
            continue
        observed = group[group['time'].dt.tz_localize(None)<end]
        column = 'location' if 'location' in group else 'location_id'
        for identity,count in observed[column].value_counts().items():
            weights[vocabulary[identity]] += count
    if (weights>0).sum()<2:
        raise ValueError('ATSTLSTM pairwise ranking needs at least two locations observed during training')
    metadata = result.metadata
    result.metadata = replace(metadata,info={**metadata.info,'negative_weights':weights},
        input_layout={**metadata.input_layout,'time_encoding':'observed_inter_checkin_interval','time_unit':'seconds',
            'distance_unit':'WGS84_geodesic_kilometres','first_time_interval':100.,'first_distance':1.,
            'candidate_geometry':'last_observed_to_each_static_POI','query_time':'last_observation_or_explicit_known_offset',
            'popularity_fitted_on':'training_interval'})
    result.split_summary.update(candidate_geometry='last_observed_to_each_static_POI',
        popularity_fitted_on='training_interval',negative_sampling='popularity_without_replacement_excluding_positive',
        evaluation_candidates='full_location_vocabulary')
    return result
