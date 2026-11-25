import json
import torch
from importlib.resources import files
from pathlib import Path


class ConfigParser(object):
    """
    use to parse the user defined parameters and use these to modify the
    pipeline's parameter setting.
    值得注意的是，目前各阶段的参数是放置于同一个 dict 中的，因此需要编程时保证命名空间不冲突。
    config 优先级：命令行/other_args > 调参参数 > config file > default config
    """

    def __init__(self, task, model, dataset, config_file=None,
                 saved_model=None, train=None, other_args=None, hyper_config_dict=None):
        """
        Args:
            task, model, dataset (str): 用户在命令行必须指明的三个参数
            config_file (str): 配置文件的文件名，将在项目根目录下进行搜索
            other_args (dict): 通过命令行传入的其他参数
        """
        self.config = {}
        self._parse_external_config(task, model, dataset, saved_model, train, other_args, hyper_config_dict)
        self._parse_config_file(config_file)
        self._load_default_config()
        # None means the caller did not explicitly choose a mode. Defaults are
        # filled only after user JSON and component/data configurations are read.
        for name in ('train', 'saved_model'):
            if name not in self.config:
                self.config[name] = True
            elif not isinstance(self.config[name], bool):
                raise TypeError(f'{name} must be a JSON boolean or an explicit boolean argument.')
        if self.config['task'] == 'map_matching':
            self.config['train'] = False
        self._init_device()

    def _parse_external_config(self, task, model, dataset,
                               saved_model=None, train=None, other_args=None, hyper_config_dict=None):
        if task is None:
            raise ValueError('the parameter task should not be None!')
        if model is None:
            raise ValueError('the parameter model should not be None!')
        if dataset is None:
            raise ValueError('the parameter dataset should not be None!')
        # 目前暂定这三个参数必须由用户指定
        self.config['task'] = task
        self.config['model'] = model
        self.config['dataset'] = dataset
        if hyper_config_dict is not None:
            # 超参数调整时传入的待调整的参数，优先级低于命令行参数
            for key in hyper_config_dict:
                self.config[key] = hyper_config_dict[key]
        if saved_model is not None:
            self.config['saved_model'] = saved_model
        if train is not None:
            self.config['train'] = train
        if other_args is not None:
            for key in other_args:
                if key not in ('train', 'saved_model') or other_args[key] is not None:
                    self.config[key] = other_args[key]

    def _parse_config_file(self, config_file):
        if config_file is not None:
            # TODO: 对 config file 的格式进行检查
            config_path = Path(config_file)
            if not config_path.name.endswith('.json'):
                config_path = config_path.with_name(config_path.name + '.json')
            if config_path.is_file():
                with config_path.open('r', encoding='utf-8') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                raise FileNotFoundError(f'Config file {config_path} is not found.')

    def _load_default_config(self):
        config_files = files(__package__)
        # 首先加载 task config
        with config_files.joinpath('task_config.json').open('r', encoding='utf-8') as f:
            task_config = json.load(f)
            if self.config['task'] not in task_config:
                raise ValueError(
                    'task {} is not supported.'.format(self.config['task']))
            task_config = task_config[self.config['task']]
            # check model and dataset
            if self.config['model'] not in task_config['allowed_model']:
                raise ValueError('task {} do not support model {}'.format(
                    self.config['task'], self.config['model']))
            model = self.config['model']
            # 加载 dataset、executor、evaluator 的模块
            if 'dataset_class' not in self.config:
                self.config['dataset_class'] = task_config[model]['dataset_class']
            if self.config['task'] == 'traj_loc_pred' and 'traj_encoder' not in self.config:
                self.config['traj_encoder'] = task_config[model]['traj_encoder']
            if self.config['task'] == 'eta' and 'eta_encoder' not in self.config:
                self.config['eta_encoder'] = task_config[model]['eta_encoder']
            if 'executor' not in self.config:
                self.config['executor'] = task_config[model]['executor']
            if 'evaluator' not in self.config:
                self.config['evaluator'] = task_config[model]['evaluator']
            # 对于 LSTM RNN GRU 使用的都是同一个类，只是 RNN 模块不一样而已，这里做一下修改
            if self.config['model'].upper() in ['LSTM', 'GRU', 'RNN']:
                self.config['rnn_type'] = self.config['model']
                self.config['model'] = 'RNN'
            # if self.config['dataset'] not in task_config['allowed_dataset']:
            #     raise ValueError('task {} do not support dataset {}'.format(
            #         self.config['task'], self.config['dataset']))
        # 接着加载每个阶段的 default config
        default_file_list = []
        # model
        default_file_list.append('model/{}/{}.json'.format(self.config['task'], self.config['model']))
        # dataset
        default_file_list.append('data/{}.json'.format(self.config['dataset_class']))
        # executor
        default_file_list.append('executor/{}.json'.format(self.config['executor']))
        # evaluator
        default_file_list.append('evaluator/{}.json'.format(self.config['evaluator']))
        # 加载所有默认配置
        for file_name in default_file_list:
            with config_files.joinpath(file_name).open('r', encoding='utf-8') as f:
                x = json.load(f)
                for key in x:
                    if key not in self.config:
                        self.config[key] = x[key]
        # 加载数据集config.json
        with open('./raw_data/{}/config.json'.format(self.config['dataset']), 'r') as f:
            x = json.load(f)
            for key in x:
                if key == 'info':
                    for ik in x[key]:
                        if ik not in self.config:
                            self.config[ik] = x[key][ik]
                else:
                    if key not in self.config:
                        self.config[key] = x[key]

    def _init_device(self):
        use_gpu = self.config.get('gpu', True)
        gpu_id = self.config.get('gpu_id', 0)
        if use_gpu and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        self.config['device'] = torch.device(
            "cuda:%d" % gpu_id if torch.cuda.is_available() and use_gpu else "cpu")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    # 支持迭代操作
    def __iter__(self):
        return self.config.__iter__()
