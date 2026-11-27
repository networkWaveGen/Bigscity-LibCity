"""Static POI categories and a streamed subset of explicit pretrained vectors."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from libcity.data.task_bundles import atomic,prepare_personalized_trajectory


def load_vectors(path,wanted):
    """Read GloVe text without keeping vectors unused by the static POI profile."""
    selected,dimension = {},None
    with path.open(encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            fields = line.split()
            if not fields:
                continue
            first = dimension is None
            if first:
                dimension = len(fields)-1
                if dimension<1:
                    raise ValueError(f'SERM word vector file has no vector columns at line {number}')
            if fields[0] in wanted or first:
                try:
                    vector = np.asarray(fields[1:],dtype=np.float32)
                except ValueError as error:
                    raise ValueError(f'SERM word vector file has invalid numbers at line {number}') from error
                if vector.shape!=(dimension,) or not np.isfinite(vector).all():
                    raise ValueError(f'SERM word vector file has inconsistent or nonfinite vectors at line {number}')
                if fields[0] in wanted and fields[0] not in selected:
                    selected[fields[0]] = vector
            if len(selected)==len(wanted):
                break
    if dimension is None:
        raise ValueError('SERM word vector file must not be empty')
    return selected,dimension


def prepare_semantic(data,training,evaluation,seed,logger=None):
    result = prepare_personalized_trajectory(data,training,evaluation,seed,logger)
    folder,raw = atomic(data)
    column = raw.get('semantic_column','venue_category_name')
    vector_file = raw.get('word_vector_file')
    if not isinstance(vector_file,str) or not vector_file:
        raise ValueError('SERM requires word_vector_file in the dataset config.json')
    path = Path(vector_file)
    path = (folder/path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'SERM word vector file does not exist: {path}')
    geo = pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    if not isinstance(column,str) or column not in geo:
        raise ValueError('SERM requires the declared semantic_column in .geo')
    # The same .geo order defines location IDs in the common trajectory adapter.
    categories = [str(value).lower().split() if not pd.isna(value) else [] for value in geo[column]]
    wanted = {word for category in categories for word in category}
    selected,dimension = load_vectors(path,wanted)
    words = sorted(selected)
    vocabulary = {word:index+1 for index,word in enumerate(words)}
    encoded = [[vocabulary[word] for word in category if word in vocabulary] for category in categories]
    width = max(1,max(map(len,encoded),default=0))
    tokens = np.zeros((len(categories)+1,width),dtype=np.int64)
    for index,values in enumerate(encoded,1):
        tokens[index,:len(values)] = values
    vectors = np.zeros((len(words)+1,dimension),dtype=np.float32)
    for index,word in enumerate(words,1):
        vectors[index] = selected[word]
    metadata = result.metadata
    result.metadata = replace(metadata,info={**metadata.info,'word_vectors':vectors,'poi_word_ids':tokens,
        'word_vocabulary':['<padding>',*words],'semantic_column':column},
        input_layout={**metadata.input_layout,'semantic_source':'static_POI_categories',
            'semantic_encoding':'lowercase_whitespace_mean_of_known_words','empty_semantics':'zero_vector'})
    result.split_summary.update(semantic_source='static_POI_categories',semantic_column=column,
        semantic_known_words=len(words),semantic_word_dim=dimension,semantic_empty_pois=sum(not row for row in encoded),
        word_vector_file=str(path))
    return result
