"""Static geographic words and canonical target-time trajectory samples."""
from dataclasses import replace

import numpy as np
import torch

from libcity.data.quadkeys import QuadkeyVocabulary,quadkey,quadkey_ngrams
from libcity.data.task_bundles import atomic,prepare_personalized_trajectory,sessions
from libcity.data.trajectory_points import read_points


def prepare_geosan(data,training,evaluation,seed,logger=None):
    result = prepare_personalized_trajectory(data,training,evaluation,seed,logger)
    folder,raw = atomic(data)
    _,coordinates = read_points(folder,raw,data['dataset'],result.metadata.num_nodes,'GeoSAN')
    words = [quadkey_ngrams(quadkey(*point,data['quadkey_level']),data['quadkey_ngram']) for point in coordinates]
    vocabulary = QuadkeyVocabulary()
    vocabulary.build_vocab(words)
    codes = np.zeros((len(words)+1,len(words[0])),dtype=np.int64)
    codes[1:] = vocabulary.numericalize(words).numpy()
    groups,cuts = sessions(folder,raw,data)
    info = result.metadata.info
    users = {identity:index+1 for index,identity in enumerate(info['user_ids'])}
    locations = {identity:index for index,identity in enumerate(info['location_ids'])}
    visited = [set() for _ in range(info['uid_size']+1)]
    prefixes = {}
    for group in groups:
        if len(group)<data['min_session_length']:
            continue
        time = group['time']
        calendar = torch.tensor((24*time.dt.day_of_week+time.dt.hour+1).to_numpy(),dtype=torch.long)
        column = 'location' if 'location' in group else 'location_id'
        for location,instant in zip(group[column],time):
            if instant.value<int(cuts[0].astype('int64')):
                visited[users[group['entity_id'].iloc[0]]].add(locations[location])
        for position in range(1,len(group)):
            prefixes[int(group['dyna_id'].iloc[position])] = calendar[:position]
    for loader in (result.train,result.val,result.test):
        for row in loader.dataset.rows:
            row['current_tim'] = prefixes[int(row['sample_id'])]
            row['uid'] = row['uid']+1
    maximum = max(int(row['lengths']) for loader in (result.train,result.val,result.test) for row in loader.dataset.rows)
    result.metadata = replace(result.metadata,info={**info,'uid_size':info['uid_size']+1,
        'user_ids':[None,*info['user_ids']],'tim_size':169,'tim_pad':0,'quadkey_tokens':vocabulary.itos,
        'quadkey_codes':codes,'coordinates':coordinates,'training_visited':[sorted(values) for values in visited],
        'max_prefix_length':maximum},input_layout={**result.metadata.input_layout,'time_encoding':'hour_of_week_1_to_168',
            'quadkey_level':data['quadkey_level'],'quadkey_ngram':data['quadkey_ngram'],
            'quadkey_vocabulary':'static_POIs_frequency_descending_lexical_ties_PAD_0',
            'negative_geometry':'haversine_radians_latitude_longitude',
            'visited_statistics':'atomic_training_observations','evaluation_candidates':'full_location_vocabulary'})
    result.split_summary.update(quadkey_level=data['quadkey_level'],quadkey_ngram=data['quadkey_ngram'],
        quadkey_vocabulary='static_POIs',quadkey_tokens=len(vocabulary.itos),
        visited_statistics='atomic_training_observations',max_prefix_length=maximum,
        evaluation_candidates='full_location_vocabulary')
    return result
