"""Lazy model declarations for the new experiment runtime."""
from dataclasses import dataclass
from importlib import import_module


def resolve(path):
    module, name = path.rsplit(':', 1)
    return getattr(import_module(module), name)


@dataclass(frozen=True)
class ModelSpec:
    factory: str
    task: str = 'traffic_state_pred'
    step: str | None = None
    train: str | None = None
    requires_checkpoint: bool = True
    supports_resume: bool = True
    supports_validation: bool = True
    prepare_resume: str | None = None
    updates_per_epoch: str | None = None
    evaluate: str | None = None
    prepare_data: str | None = None
    loss: str | None = None
    evaluate_loader: str | None = None
    kind: str = 'neural'
    device_types: tuple = ('cpu','cuda','mps')
    protocol: str | None = None


MODELS = {
    'ToGCN': ModelSpec('libcity.model.traffic_flow_prediction.runtime_togcn:ToGCN',
        step='libcity.model.traffic_speed_prediction.runtime_seq2seq:teacher_step'),
    'SimST': ModelSpec('libcity.model.traffic_flow_prediction.runtime_simst:SimST'),
    'TESTAM': ModelSpec('libcity.model.traffic_speed_prediction.runtime_testam:TESTAM',
        step='libcity.executor.testam_step:expert_step'),
    'MegaCRN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_megacrn:MegaCRN',
        step='libcity.model.traffic_speed_prediction.runtime_megacrn:memory_step'),
    'ASTGCN': ModelSpec('libcity.model.traffic_flow_prediction.runtime_multiscale:ASTGCN'),
    'MSTGCN': ModelSpec('libcity.model.traffic_flow_prediction.runtime_multiscale:MSTGCN'),
    'ChebConv': ModelSpec('libcity.model.road_representation.runtime_reconstruction:ChebConv',task='road_representation',
        protocol='road_reconstruction',prepare_data='libcity.data.road_reconstruction:prepare_reconstruction',
        loss='libcity.executor.road_reconstruction:ReconstructionLoss',
        evaluate_loader='libcity.executor.road_reconstruction:evaluate_reconstruction',device_types=('cpu','cuda')),
    'GAT': ModelSpec('libcity.model.road_representation.runtime_reconstruction:GAT',task='road_representation',
        protocol='road_reconstruction',prepare_data='libcity.data.road_reconstruction:prepare_reconstruction',
        loss='libcity.executor.road_reconstruction:ReconstructionLoss',
        evaluate_loader='libcity.executor.road_reconstruction:evaluate_reconstruction'),
    'FPMC': ModelSpec('libcity.model.trajectory_loc_prediction.runtime_fpmc:FPMC', task='traj_loc_pred',
        prepare_data='libcity.data.task_bundles:prepare_personalized_trajectory',
        loss='libcity.executor.task_evaluation:RankingLoss', evaluate_loader='libcity.executor.task_evaluation:evaluate_task'),
    'STMatching': ModelSpec('libcity.model.map_matching.runtime_matching:create_stmatching', task='map_matching',
        requires_checkpoint=False, supports_resume=False, supports_validation=False,
        evaluate='libcity.model.map_matching.runtime_hmmm:evaluate', kind='direct', device_types=('cpu',)),
    'IVMM': ModelSpec('libcity.model.map_matching.runtime_matching:create_ivmm', task='map_matching',
        requires_checkpoint=False, supports_resume=False, supports_validation=False,
        evaluate='libcity.model.map_matching.runtime_hmmm:evaluate', kind='direct', device_types=('cpu',)),
    'STID': ModelSpec('libcity.model.traffic_speed_prediction.runtime_identity:STID'),
    'STAEformer': ModelSpec('libcity.model.traffic_speed_prediction.runtime_identity:STAEformer'),
    'FreTS': ModelSpec('libcity.model.traffic_flow_prediction.runtime_frets:FreTS'),
    'STNorm': ModelSpec('libcity.model.traffic_flow_prediction.runtime_stnorm:STNorm'),
    'ATDM': ModelSpec('libcity.model.traffic_speed_prediction.runtime_baselines:ATDM'),
    'AutoEncoder': ModelSpec('libcity.model.traffic_speed_prediction.runtime_baselines:AutoEncoder'),
    'AGCRN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_baselines:AGCRN'),
    'TGCN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_baselines:TGCN',
        step='libcity.model.traffic_speed_prediction.runtime_baselines:tgcn_step'),
    'TrafficRNN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_rnn:AutoregressiveRNN',
        step='libcity.model.traffic_speed_prediction.runtime_seq2seq:teacher_step'),
    'TrafficLSTM': ModelSpec('libcity.model.traffic_speed_prediction.runtime_rnn:AutoregressiveRNN',
        step='libcity.model.traffic_speed_prediction.runtime_seq2seq:teacher_step'),
    'Seq2Seq': ModelSpec('libcity.model.traffic_speed_prediction.runtime_seq2seq:Seq2Seq',
        step='libcity.model.traffic_speed_prediction.runtime_seq2seq:teacher_step'),
    'CRANN': ModelSpec('libcity.model.traffic_flow_prediction.runtime_crann:CRANN'),
    'FNN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_standard:FNN'),
    'GWNET': ModelSpec('libcity.model.traffic_speed_prediction.runtime_standard:GWNET'),
    'STSSL': ModelSpec('libcity.model.traffic_flow_prediction.runtime_stssl:STSSL',
        train='libcity.executor.stssl_train:train'),
    'RNN': ModelSpec('libcity.model.trajectory_loc_prediction.runtime_rnn:RNN',task='traj_loc_pred',
        prepare_data='libcity.data.task_bundles:prepare_trajectory',loss='libcity.executor.task_evaluation:RankingLoss',
        evaluate_loader='libcity.executor.task_evaluation:evaluate_task'),
    'DeepTTE': ModelSpec('libcity.model.eta.runtime_deeptte:DeepTTE',task='eta',
        prepare_data='libcity.data.task_bundles:prepare_eta',loss='libcity.executor.task_evaluation:ETALoss',
        step='libcity.executor.task_evaluation:eta_step',evaluate_loader='libcity.executor.task_evaluation:evaluate_task'),
    'LINE': ModelSpec('libcity.model.road_representation.runtime_line:LINE',task='road_representation',
        prepare_data='libcity.data.task_bundles:prepare_road',loss='libcity.executor.task_evaluation:EdgeLoss',
        evaluate_loader='libcity.executor.task_evaluation:evaluate_task'),
    'GRU': ModelSpec('libcity.model.traffic_speed_prediction.runtime_rnn:GRU'),
    'STGCN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_stgcn:STGCN'),
    'DCRNN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_dcrnn:DCRNN',
                       step='libcity.model.traffic_speed_prediction.runtime_dcrnn:curriculum_step'),
    'Trafformer': ModelSpec('libcity.model.traffic_speed_prediction.runtime_trafformer:Trafformer'),
    'ASTGNN': ModelSpec('libcity.model.traffic_flow_prediction.runtime_astgnn:ASTGNN',
        train='libcity.executor.astgnn_train:train',prepare_resume='libcity.executor.astgnn_train:prepare_resume'),
    'MTGNN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_mtgnn:MTGNN',
        train='libcity.executor.mtgnn_train:train',updates_per_epoch='libcity.executor.mtgnn_train:updates_per_epoch'),
    'HMMM': ModelSpec('libcity.model.map_matching.runtime_hmmm:create_model',task='map_matching',
        requires_checkpoint=False,supports_resume=False,supports_validation=False,evaluate='libcity.model.map_matching.runtime_hmmm:evaluate',
        kind='direct',device_types=('cpu',)),
}


def register_model(name, spec):
    if name in MODELS:
        raise ValueError(f'Model {name} is already registered')
    MODELS[name] = spec


def model_spec(name):
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f'Model {name!r} is not migrated; available: {sorted(MODELS)}') from None
