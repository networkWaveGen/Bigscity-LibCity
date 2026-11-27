"""Causal session boundaries, training-only time similarity and spatial parents."""
from bisect import bisect_left
from dataclasses import replace

import numpy as np
import torch

from libcity.data.task_bundles import atomic,prepare_history_trajectory,sessions
from libcity.data.trajectory_points import read_points,spherical_distance


def spatial_parents(coordinates):
    """Nearest earlier point, with the original latest-earlier tie rule (1 m)."""
    parents = np.zeros(len(coordinates),dtype=np.int64)
    for position in range(1,len(coordinates)):
        distances = np.round(spherical_distance(coordinates[position],coordinates[:position]),3)
        parents[position] = position-1-int(distances[::-1].argmin())
    return torch.from_numpy(parents)


def time_similarity(groups,train_end):
    # Fit once on atomic training observations, not on duplicated prefix rows.
    visited = [set() for _ in range(48)]
    observations = 0
    for group,locations in groups:
        time = group['time']
        codes = (time.dt.hour+24*(time.dt.day_of_week>=5)).to_numpy()
        for location,code,instant in zip(locations,codes,time):
            if instant.value<train_end:
                visited[code].add(int(location))
                observations += 1
    result = np.zeros((49,49),dtype=np.float32)
    for left in range(48):
        result[left,left] = 1.
        for right in range(left):
            union = visited[left]|visited[right]
            value = len(visited[left]&visited[right])/len(union) if union else 0.
            result[left,right] = result[right,left] = value
    return result,observations


def prepare_lstpm(data,training,evaluation,seed,logger=None):
    result = prepare_history_trajectory(data,training,evaluation,seed,logger)
    folder,raw = atomic(data)
    _,coordinates = read_points(folder,raw,data['dataset'],result.metadata.num_nodes,'LSTPM')
    groups,cuts = sessions(folder,raw,data)
    vocabulary = {identity:index for index,identity in enumerate(result.metadata.info['location_ids'])}
    encoded,users = [],{}
    for group in groups:
        if len(group)<data['min_session_length']:
            continue
        column = 'location' if 'location' in group else 'location_id'
        locations = np.asarray([vocabulary[value] for value in group[column]],dtype=np.int64)
        encoded.append((group,locations))
        users.setdefault(group['entity_id'].iloc[0],[]).append(
            (group['time'].iloc[-1].value,len(group),coordinates[locations].mean(axis=0)))
    banks = {}
    for user,entries in users.items():
        entries.sort(key=lambda entry:entry[0])
        banks[user] = ([entry[0] for entry in entries],
            torch.tensor([entry[1] for entry in entries]),np.stack([entry[2] for entry in entries]))
    features = {}
    for group,locations in encoded:
        endings,lengths,centroids = banks[group['entity_id'].iloc[0]]
        stop = bisect_left(endings,group['time'].iloc[0].value)
        previous = lengths[:stop]
        # One causal parent bank per trip; all shorter prefixes share its views.
        parents = spatial_parents(coordinates[locations[:-1]])
        distances = torch.tensor(np.maximum(1.,np.round(
            spherical_distance(coordinates[locations[:-1],None],centroids[None,:stop]),3)),dtype=torch.float32)
        for position in range(1,len(group)):
            features[int(group['dyna_id'].iloc[position])] = (previous,parents[:position],distances[position-1])
    for loader in (result.train,result.val,result.test):
        for row in loader.dataset.rows:
            lengths,parents,distances = features[int(row['sample_id'])]
            if int(lengths.sum())!=int(row['history_lengths']):
                raise ValueError('LSTPM session boundaries disagree with the completed history bank')
            row.update(history_session_lengths=lengths,dilated_rnn_input_index=parents,history_avg_distance=distances)
    similarity,observations = time_similarity(encoded,int(cuts[0].astype('int64')))
    metadata = result.metadata
    result.metadata = replace(metadata,info={**metadata.info,'tim_sim_matrix':similarity},input_layout={
        **metadata.input_layout,'history_sessions':'explicit_lengths_with_carried_LSTM_state',
        'time_similarity_fitted_on':'atomic_training_observations',
        'spatial_parents':'nearest_earlier_point_latest_tie',
        'distance_unit':'great_circle_6371_kilometres_rounded_3_decimals',
        'history_distance':'last_observed_POI_to_arithmetic_session_centroid_min_1_km'})
    result.split_summary.update(time_similarity_fitted_on='atomic_training_observations',
        time_similarity_observations=observations,spatial_parent_ties='latest_earlier_point',
        history_distance='arithmetic_session_centroid_min_1_km')
    return result
