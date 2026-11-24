"""Map task configuration names to their implementation modules."""

from importlib import import_module


MODEL_MODULES = {
    'traffic_state_pred': {
        'ACFM': 'model.traffic_flow_prediction.ACFM',
        'ACFMCommon': 'model.traffic_flow_prediction.ACFMCommon',
        'AGCRN': 'model.traffic_flow_prediction.AGCRN',
        'ASTGCN': 'model.traffic_flow_prediction.ASTGCN',
        'ASTGCNCommon': 'model.traffic_flow_prediction.ASTGCNCommon',
        'ASTGNN': 'model.traffic_flow_prediction.ASTGNN',
        'ASTGNNCommon': 'model.traffic_flow_prediction.ASTGNNCommon',
        'ATDM': 'model.traffic_speed_prediction.ATDM',
        'AutoEncoder': 'model.traffic_speed_prediction.AutoEncoder',
        'CCRNN': 'model.traffic_demand_prediction.CCRNN',
        'CONVGCN': 'model.traffic_flow_prediction.CONVGCN',
        'CONVGCNCommon': 'model.traffic_flow_prediction.CONVGCNCommon',
        'CRANN': 'model.traffic_flow_prediction.CRANN',
        'CSTN': 'model.traffic_od_prediction.CSTN',
        'D2STGNN': 'model.traffic_speed_prediction.D2STGNN',
        'DCRNN': 'model.traffic_speed_prediction.DCRNN',
        'DGCN': 'model.traffic_flow_prediction.DGCN',
        'DKFN': 'model.traffic_speed_prediction.DKFN',
        'DMSTGCN': 'model.traffic_speed_prediction.DMSTGCN',
        'DMVSTNet': 'model.traffic_demand_prediction.DMVSTNet',
        'DSAN': 'model.traffic_flow_prediction.DSAN',
        'DSTAGNN': 'model.traffic_flow_prediction.DSTAGNN',
        'ESG': 'model.traffic_flow_prediction.ESG',
        'FNN': 'model.traffic_speed_prediction.FNN',
        'FOGS': 'model.traffic_flow_prediction.FOGS',
        'FreTS': 'model.traffic_flow_prediction.FreTS',
        'GEML': 'model.traffic_od_prediction.GEML',
        'GMAN': 'model.traffic_speed_prediction.GMAN',
        'GSNet': 'model.traffic_accident_prediction.GSNet',
        'GTS': 'model.traffic_speed_prediction.GTS',
        'GWNET': 'model.traffic_speed_prediction.GWNET',
        'HGCN': 'model.traffic_speed_prediction.HGCN',
        'HIEST': 'model.traffic_speed_prediction.HIEST',
        'MSTGCN': 'model.traffic_flow_prediction.MSTGCN',
        'MSTGCNCommon': 'model.traffic_flow_prediction.MSTGCNCommon',
        'MTGNN': 'model.traffic_speed_prediction.MTGNN',
        'MegaCRN': 'model.traffic_speed_prediction.MegaCRN',
        'MultiSPANS': 'model.traffic_flow_prediction.MultiSPANS',
        'MultiSTGCnet': 'model.traffic_flow_prediction.MultiSTGCnet',
        'MultiSTGCnetCommon': 'model.traffic_flow_prediction.MultiSTGCnetCommon',
        'PDFormer': 'model.traffic_flow_prediction.PDFormer',
        'RGSL': 'model.traffic_flow_prediction.RGSL',
        'RNN': 'model.traffic_speed_prediction.RNN',
        'ResLSTM': 'model.traffic_flow_prediction.ResLSTM',
        'SSTBAN': 'model.traffic_flow_prediction.SSTBAN',
        'STAEformer': 'model.traffic_speed_prediction.STAEformer',
        'STAGGCN': 'model.traffic_speed_prediction.STAGGCN',
        'STDN': 'model.traffic_flow_prediction.STDN',
        'STG2Seq': 'model.traffic_demand_prediction.STG2Seq',
        'STGCN': 'model.traffic_speed_prediction.STGCN',
        'STGNCDE': 'model.traffic_flow_prediction.STGNCDE',
        'STGODE': 'model.traffic_flow_prediction.STGODE',
        'STID': 'model.traffic_speed_prediction.STID',
        'STMGAT': 'model.traffic_speed_prediction.STMGAT',
        'STNN': 'model.traffic_flow_prediction.STNN',
        'STNorm': 'model.traffic_flow_prediction.STNorm',
        'STPGCN': 'model.traffic_flow_prediction.STPGCN',
        'STResNet': 'model.traffic_flow_prediction.STResNet',
        'STResNetCommon': 'model.traffic_flow_prediction.STResNetCommon',
        'STSGCN': 'model.traffic_flow_prediction.STSGCN',
        'STSSL': 'model.traffic_flow_prediction.STSSL',
        'STTN': 'model.traffic_speed_prediction.STTN',
        'STTSNet': 'model.traffic_flow_prediction.STTSNet',
        'STWave': 'model.traffic_flow_prediction.STWave',
        'Seq2Seq': 'model.traffic_speed_prediction.Seq2Seq',
        'SimST': 'model.traffic_flow_prediction.SimST',
        'TESTAM': 'model.traffic_speed_prediction.TESTAM',
        'TGCLSTM': 'model.traffic_speed_prediction.TGCLSTM',
        'TGCN': 'model.traffic_speed_prediction.TGCN',
        'TemplateTSP': 'model.traffic_speed_prediction.TemplateTSP',
        'TimeMixer': 'model.traffic_flow_prediction.TimeMixer',
        'ToGCN': 'model.traffic_flow_prediction.ToGCN',
        'Trafformer': 'model.traffic_speed_prediction.Trafformer',
    },
    'traj_loc_pred': {
        'ATSTLSTM': 'model.trajectory_loc_prediction.ATSTLSTM',
        'CARA': 'model.trajectory_loc_prediction.CARA',
        'DeepMove': 'model.trajectory_loc_prediction.DeepMove',
        'FPMC': 'model.trajectory_loc_prediction.FPMC',
        'GeoSAN': 'model.trajectory_loc_prediction.GeoSAN',
        'HSTLSTM': 'model.trajectory_loc_prediction.HSTLSTM',
        'LSTPM': 'model.trajectory_loc_prediction.LSTPM',
        'RNN': 'model.trajectory_loc_prediction.RNN',
        'SERM': 'model.trajectory_loc_prediction.SERM',
        'STAN': 'model.trajectory_loc_prediction.STAN',
        'STRNN': 'model.trajectory_loc_prediction.STRNN',
        'TemplateTLP': 'model.trajectory_loc_prediction.TemplateTLP',
    },
    'eta': {
        'DeepTTE': 'model.eta.DeepTTE',
        'TTPNet': 'model.eta.TTPNet',
    },
    'map_matching': {
        'HMMM': 'model.map_matching.HMMM',
        'IVMM': 'model.map_matching.IVMM',
        'STMatching': 'model.map_matching.STMatching',
    },
    'road_representation': {
        'ChebConv': 'model.road_representation.ChebConv',
        'DeepWalk': 'model.road_representation.DeepWalk',
        'GAT': 'model.road_representation.GAT',
        'GeomGCN': 'model.road_representation.GeomGCN',
        'LINE': 'model.road_representation.LINE',
        'Node2Vec': 'model.road_representation.Node2Vec',
    },
}

COMPONENT_MODULES = {
    'dataset_class': {
        'ACFMDataset': 'data.dataset.dataset_subclass.acfm_dataset',
        'ASTGCNDataset': 'data.dataset.dataset_subclass.astgcn_dataset',
        'AbstractDataset': 'data.dataset.abstract_dataset',
        'CCRNNDataset': 'data.dataset.dataset_subclass.ccrnn_dataset',
        'CONVGCNDataset': 'data.dataset.dataset_subclass.convgcn_dataset',
        'CRANNDataset': 'data.dataset.dataset_subclass.crann_dataset',
        'CSTNDataset': 'data.dataset.dataset_subclass.cstn_dataset',
        'ChebConvDataset': 'data.dataset.dataset_subclass.chebconv_dataset',
        'DMSTGCNDataset': 'data.dataset.dataset_subclass.dmstgcn_dataset',
        'DMVSTNetDataset': 'data.dataset.dataset_subclass.dmvstnet_dataset',
        'DSTAGNNDataset': 'data.dataset.dataset_subclass.dstagnn_dataset',
        'ESGDataset': 'data.dataset.dataset_subclass.esg_dataset',
        'ETADataset': 'data.dataset.eta_dataset',
        'FOGSDataset': 'data.dataset.dataset_subclass.fogs_dataset',
        'GMANDataset': 'data.dataset.dataset_subclass.gman_dataset',
        'GSNetDataset': 'data.dataset.dataset_subclass.gsnet_dataset',
        'GTSDataset': 'data.dataset.dataset_subclass.gts_dataset',
        'GeoSANDataset': 'data.dataset.dataset_subclass.geosan_dataset',
        'HGCNDataset': 'data.dataset.dataset_subclass.hgcn_dataset',
        'HIESTDataset': 'data.dataset.dataset_subclass.hiest_dataset',
        'LINEDataset': 'data.dataset.dataset_subclass.line_dataset',
        'MapMatchingDataset': 'data.dataset.map_matching_dataset',
        'MultiSTGCnetDataset': 'data.dataset.dataset_subclass.multi_stgcnet_dataset',
        'PBSTrajectoryDataset': 'data.dataset.dataset_subclass.pbs_trajectory_dataset',
        'PDFormerDataset': 'data.dataset.dataset_subclass.pdformer_dataset',
        'RESLSTMDataset': 'data.dataset.dataset_subclass.reslstm_dataset',
        'RoadNetWorkDataset': 'data.dataset.roadnetwork_dataset',
        'SSTBANDataset': 'data.dataset.dataset_subclass.sstban_dataset',
        'STAEformerDataset': 'data.dataset.dataset_subclass.staeformer_dataset',
        'STAGGCNDataset': 'data.dataset.dataset_subclass.staggcn_dataset',
        'STDNDataset': 'data.dataset.dataset_subclass.stdn_dataset',
        'STG2SeqDataset': 'data.dataset.dataset_subclass.stg2seq_dataset',
        'STGODEDataset': 'data.dataset.dataset_subclass.stgode_dataset',
        'STPGCNDataset': 'data.dataset.dataset_subclass.stpgcn_dataset',
        'STResNetDataset': 'data.dataset.dataset_subclass.stresnet_dataset',
        'STTSNetDataset': 'data.dataset.dataset_subclass.sttsnet_dataset',
        'STWaveDataset': 'data.dataset.dataset_subclass.stwave_dataset',
        'SimSTDataset': 'data.dataset.dataset_subclass.simst_dataset',
        'TGCLSTMDataset': 'data.dataset.dataset_subclass.tgclstm_dataset',
        'TrafficStateCPTDataset': 'data.dataset.traffic_state_cpt_dataset',
        'TrafficStateDataset': 'data.dataset.traffic_state_datatset',
        'TrafficStateGridDataset': 'data.dataset.traffic_state_grid_dataset',
        'TrafficStateGridOdDataset': 'data.dataset.traffic_state_grid_od_dataset',
        'TrafficStateOdDataset': 'data.dataset.traffic_state_od_dataset',
        'TrafficStatePointDataset': 'data.dataset.traffic_state_point_dataset',
        'TrajectoryDataset': 'data.dataset.trajectory_dataset',
    },
    'executor': {
        'ASTGNNExecutor': 'executor.astgnn_executor',
        'AbstractTraditionExecutor': 'executor.abstract_tradition_executor',
        'ChebConvExecutor': 'executor.chebconv_executor',
        'DCRNNExecutor': 'executor.dcrnn_executor',
        'ETAExecutor': 'executor.eta_executor',
        'FOGSExecutor': 'executor.fogs_executor',
        'GEMLExecutor': 'executor.geml_executor',
        'GensimExecutor': 'executor.gensim_executor',
        'GeoSANExecutor': 'executor.geosan_executor',
        'HyperTuning': 'executor.hyper_tuning',
        'LINEExecutor': 'executor.line_executor',
        'MTGNNExecutor': 'executor.mtgnn_executor',
        'MapMatchingExecutor': 'executor.map_matching_executor',
        'MegaCRNExecutor': 'executor.megacrn_executor',
        'PDFormerExecutor': 'executor.pdformer_executor',
        'SSTBANExecutor': 'executor.sstban_executor',
        'STSSLExecutor': 'executor.STSSL_executor',
        'STTSNetExecutor': 'executor.sttsnet_executor',
        'TESTAMExecutor': 'executor.testam_executor',
        'TimeMixerExecutor': 'executor.timemixer_executor',
        'TrafficStateExecutor': 'executor.traffic_state_executor',
        'TrafformerExecutor': 'executor.trafformer_executor',
        'TrajLocPredExecutor': 'executor.traj_loc_pred_executor',
    },
    'evaluator': {
        'CARALocPredEvaluator': 'evaluator.cara_loc_pred_evaluator',
        'ETAEvaluator': 'evaluator.eta_evaluator',
        'GeoSANEvaluator': 'evaluator.geosan_evaluator',
        'MapMatchingEvaluator': 'evaluator.map_matching_evaluator',
        'RoadRepresentationEvaluator': 'evaluator.road_representation_evaluator',
        'TrafficAccidentEvaluator': 'evaluator.traffic_accident_evaluator',
        'TrafficStateEvaluator': 'evaluator.traffic_state_evaluator',
        'TrajLocPredEvaluator': 'evaluator.traj_loc_pred_evaluator',
    },
    'traj_encoder': {
        'AtstlstmEncoder': 'data.dataset.trajectory_encoder.atstlstm_encoder',
        'CARATrajectoryEncoder': 'data.dataset.trajectory_encoder.cara_encoder',
        'HstlstmEncoder': 'data.dataset.trajectory_encoder.hstlstm_encoder',
        'LstpmEncoder': 'data.dataset.trajectory_encoder.lstpm_encoder',
        'SermEncoder': 'data.dataset.trajectory_encoder.serm_encoder',
        'StanEncoder': 'data.dataset.trajectory_encoder.stan_encoder',
        'StandardTrajectoryEncoder': 'data.dataset.trajectory_encoder.standard_trajectory_encoder',
        'StrnnEncoder': 'data.dataset.trajectory_encoder.strnn_encoder',
    },
    'eta_encoder': {
        'DeeptteEncoder': 'data.dataset.eta_encoder.deeptte_encoder',
        'TtpnetEncoder': 'data.dataset.eta_encoder.ttpnet_encoder',
    },
}


def component_module(kind, name, *, task=None):
    """Return the implementation module for a registered component."""
    if kind == 'model':
        if task not in MODEL_MODULES:
            raise ValueError(f'Unsupported task: {task}')
        modules = MODEL_MODULES[task]
    else:
        modules = COMPONENT_MODULES[kind]
    try:
        return 'libcity.' + modules[name]
    except KeyError:
        raise AttributeError(f'{kind} {name!r} is not registered') from None


def load_component(kind, name, *, task=None):
    """Return the registered class for the given name."""
    return getattr(import_module(component_module(kind, name, task=task)), name)


def get_executor(config, model, data_feature):
    executor = load_component('executor', config['executor'])
    return executor(config, model, data_feature)


def get_model(config, data_feature):
    model = load_component('model', config['model'], task=config['task'])
    return model(config, data_feature)


def get_evaluator(config):
    evaluator = load_component('evaluator', config['evaluator'])
    return evaluator(config)
