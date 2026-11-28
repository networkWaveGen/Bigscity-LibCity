"""Two observed streams with train-only scaling and historical calendar."""
from dataclasses import replace

from torch.utils.data import DataLoader,Dataset

from libcity.data.calendar_bundle import prepare_calendar
from libcity.data.timeline import boundaries
from libcity.data.traffic_bundle import fit_scaler


class ScaledCalendarWindows(Dataset):
    def __init__(self,calendar,scalers):
        self.calendar,self.scalers = calendar,scalers

    def __len__(self):
        return len(self.calendar)

    def __getitem__(self,index):
        sample = self.calendar[index]
        for channel,scaler in self.scalers.items():
            sample['X'][...,channel] = scaler.transform(sample['X'][...,channel])
        return sample


def prepare_dmstgcn(data,training,evaluation,seed,logger=None):
    features = data['features']
    primary,auxiliary = features['primary_index'],features['auxiliary_index']
    plain = {**data,'graph':dict(name='none',symmetric=False,self_loops=False),
        'features':{key:value for key,value in features.items() if key not in ('primary_index','auxiliary_index')}}
    bundle = prepare_calendar(plain,training,evaluation,seed,logger)
    meta = bundle.metadata
    observed = meta.input_layout['observed_feature_indices']
    if primary==auxiliary or primary not in observed or auxiliary not in observed:
        raise ValueError('DMSTGCN requires two distinct historical observation indices, excluding known calendar/future features')
    raw = bundle.train.dataset.windows
    end = boundaries(len(raw.clock),data['split'],raw.clock)[0]
    scalers = {index:fit_scaler(data['scaler']['name'],raw.history[:end,:,index])
        for index in (primary,auxiliary) if index>=meta.output_dim}
    description = dict(primary_index=primary,auxiliary_index=auxiliary,
        extra_scalers={str(index):dict(name=type(scaler).__name__,state=vars(scaler)) for index,scaler in scalers.items()},
        scaler_fit_end_exclusive=str(raw.clock[end]),graph_time='first_observed_history_timestamp')
    for name in ('train','val','test'):
        loader = getattr(bundle,name)
        dataset = ScaledCalendarWindows(loader.dataset,scalers)
        setattr(bundle,name,DataLoader(dataset,batch_size=loader.batch_size,num_workers=loader.num_workers,
            shuffle=training['shuffle'] if name=='train' else False,generator=loader.generator))
    bundle.metadata = replace(meta,input_layout={**meta.input_layout,'dmstgcn_features':description})
    bundle.split_summary['dmstgcn_features'] = description
    return bundle
