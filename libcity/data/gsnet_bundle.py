"""Full risk grids and partial semantic graphs share explicit physical cell IDs."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from libcity.data.bundle import DatasetMetadata
from libcity.data.grid_traffic import prepare_grid
from libcity.data.timeline import boundaries
from libcity.data.traffic_bundle import fit_scaler
from libcity.utils.runtime_checkpoint import plain


def prepare_gsnet(data,training,evaluation,seed,logger=None):
    if data['sampling']['name']!='risk_cpt' or data['graph']['name']!='risk_multigraph':
        raise ValueError('GSNet data require risk_cpt sampling and risk_multigraph relations')
    state = {}
    def geometry(raw,geo):
        shape = [raw.get('grid_len_row'),raw.get('grid_len_column')]
        if any(type(value) is not int or value<1 for value in shape):
            raise ValueError('GSNet metadata require positive grid_len_row/grid_len_column')
        if not {'geo_id','row_id','column_id'}<=set(geo) or not len(geo) or geo['geo_id'].isna().any() or geo['geo_id'].duplicated().any():
            raise ValueError('GSNet .geo requires distinct graph node IDs and grid positions')
        if pd.api.types.is_numeric_dtype(geo['geo_id']) and not np.isfinite(geo['geo_id']).all():
            raise ValueError('GSNet graph node IDs must be finite')
        positions = geo[['row_id','column_id']].to_numpy(dtype=np.float64)
        if (not np.isfinite(positions).all() or np.any(positions<0) or np.any(positions!=np.floor(positions)) or
                np.any(positions[:,0]>=shape[0]) or np.any(positions[:,1]>=shape[1])):
            raise ValueError('GSNet graph positions must lie inside the declared grid')
        mask_column = raw.get('data_col_risk_mask','risk_mask')
        if mask_column not in geo:
            raise ValueError('GSNet .geo must supply a risk_mask column')
        mask = geo[mask_column].to_numpy(dtype=np.float64)
        if not np.isfinite(mask).all() or np.any(mask<0) or not np.any(mask>0):
            raise ValueError('GSNet risk_mask must be finite, nonnegative and nonempty')
        cells = (positions[:,0]*shape[1]+positions[:,1]).astype(np.int64)
        region = np.zeros(shape[0]*shape[1],dtype=np.float64)
        used = {}
        for cell,value in zip(cells,mask):
            if cell in used and used[cell]!=value:
                raise ValueError('Graph nodes in the same grid cell require the same risk_mask')
            region[cell] = value;used[cell] = value
        mapping = np.zeros((len(region),len(geo)),dtype=np.float32)
        mapping[cells,np.arange(len(geo))] = 1.
        state.update(raw=raw,ids=geo['geo_id'].tolist(),cells=cells,region=region,mapping=mapping)
        indices = np.arange(len(region))
        return pd.DataFrame(dict(geo_id=indices,row_id=indices//shape[1],column_id=indices%shape[1]))

    raw_config = dict(data,graph=dict(name='risk_multigraph'),features={key:value for key,value in data['features'].items()
        if key not in ('graph_columns','external_scaler')})
    bundle = prepare_grid(raw_config,training,evaluation,seed,logger,geo_transform=geometry)
    if bundle.metadata.output_dim!=1:
        raise ValueError('GSNet supports exactly one leading risk target')
    arrays = bundle.train.dataset.arrays
    targets = arrays['history'][...,:1]
    if np.any(targets[np.isfinite(targets)]<0):
        raise ValueError('GSNet risk observations must be nonnegative or NaN')
    raw,ids = state['raw'],state['ids']
    config = data['graph']
    columns = raw.get('weight_col',[])
    columns = [columns] if isinstance(columns,str) else columns
    selected = {}
    for index,name in enumerate(('road','risk','poi')):
        column = config.get(name+'_column')
        column = column if column is not None else columns[index] if isinstance(columns,list) and index<len(columns) else None
        if name=='poi' and not config.get('use_poi',True):
            column = None
        if column is not None and (not isinstance(column,str) or not column):
            raise ValueError('GSNet relation columns must be nonempty names')
        if name!='poi' and column is None:
            raise ValueError('GSNet requires named road and risk relation columns')
        if column is not None:
            selected[name] = column
    if len(set(selected.values()))!=len(selected):
        raise ValueError('GSNet graph views require distinct relation columns')
    path = Path(data['data_dir'])/data['dataset']/(raw.get('rel_file',data['dataset'])+'.rel')
    relation = pd.read_csv(path)
    if not {'origin_id','destination_id',*selected.values()}<=set(relation):
        raise ValueError('GSNet .rel requires graph endpoints and selected view columns')
    node_index = {value:index for index,value in enumerate(ids)}
    try:
        source = np.array([node_index[value] for value in relation['origin_id']],dtype=np.int64)
        destination = np.array([node_index[value] for value in relation['destination_id']],dtype=np.int64)
    except KeyError as error:
        raise ValueError('GSNet relation endpoint is absent from graph .geo') from error
    if len(np.unique(source*len(ids)+destination))!=len(source):
        raise ValueError('GSNet relations require unique directed graph pairs')
    static = dict(risk_mask=state['region'],grid_node_map=state['mapping'])
    for name,column in selected.items():
        values = relation[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('GSNet relation weights must be finite')
        graph = np.zeros((len(ids),len(ids)),dtype=np.float32)
        graph[source,destination] = values
        if config.get('symmetric',False): graph = np.maximum(graph,graph.T)
        if config.get('self_loops',False): np.fill_diagonal(graph,1.)
        if not np.isfinite(graph).all():
            raise ValueError('GSNet relation weights exceed float32 graph storage')
        static[name+'_adj'] = graph
    names = list(bundle.metadata.info['data_columns'])
    features = data['features'].get('graph_columns')
    features = features if features is not None else raw.get('graph_input_col') or names[:1]
    if (not isinstance(features,list) or not features or any(not isinstance(name,str) or name not in names for name in features)
            or len(set(features))!=len(features)):
        raise ValueError('GSNet graph_columns must select distinct historical observation names')
    layout = dict(bundle.metadata.input_layout,graph_input_indices=[names.index(name) for name in features],
        graph_nodes=len(ids),graph_views=list(selected),spatial_identity='row_major_grid_cells')
    # Scale numeric global covariates only, using the original training prefix.
    # Calendar one-hots and known raw time flags retain their own meanings.
    history_names = list(layout['historical_feature_names'])
    known_names = list(layout['future_feature_names'])
    calendar_width = 24*int(data['features']['time_in_day'])+7*int(data['features']['day_in_week'])
    # The first validation window may follow a file/gap boundary; use the
    # protocol boundary itself, not the first usable validation sample.
    stop = boundaries(len(arrays['clock']),data['split'],arrays['clock'])[0]
    transformed = dict(arrays,external=arrays['external'].copy(),known=arrays['known'].copy())
    descriptions = {}
    scaler_name = data['features'].get('external_scaler',{}).get('name','minmax01')
    for index in range(calendar_width,len(history_names)):
        scaler = fit_scaler(scaler_name,arrays['external'][:stop,index])
        transformed['external'][:,index] = scaler.transform(arrays['external'][:,index])
        name = history_names[index]
        if name in known_names:
            column = known_names.index(name)
            transformed['known'][:,column] = scaler.transform(arrays['known'][:,column])
        descriptions[name] = dict(name=type(scaler).__name__,state=plain(vars(scaler)))
    if not np.isfinite(transformed['external']).all() or not np.isfinite(transformed['known']).all():
        raise ValueError('GSNet external_scaler must produce finite covariates')
    for loader in (bundle.train,bundle.val,bundle.test): loader.dataset.arrays = transformed
    layout.update(external_scalers=descriptions,external_scaler_fit_end_exclusive=bundle.split_summary['train_end'])
    info = dict(bundle.metadata.info,graph_node_ids=ids,graph_columns=features,
        risk_thresholds=raw.get('risk_thresholds'),risk_weights=raw.get('risk_weights'))
    metadata = DatasetMetadata(bundle.metadata.name,bundle.metadata.num_nodes,bundle.metadata.feature_dim,1,
        info=info,input_layout=layout,static_data=static)
    summary = dict(bundle.split_summary,graph_nodes=len(ids),graph_views=list(selected),
        risk_regions=int(np.count_nonzero(state['region'])),external_scaling=scaler_name)
    if logger: logger.info('GSNet grid/graph identities: %s',summary)
    return replace(bundle,metadata=metadata,split_summary=summary)
