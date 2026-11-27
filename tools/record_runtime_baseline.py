"""Capture fixed pre-migration GRU computation; run before changing runtime code."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import numpy as np
import torch
from runtime_fixtures import atomic_fixture
from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
from libcity.model.traffic_speed_prediction.RNN import RNN
from libcity.model.loss import masked_mae_torch


def main():
    destination = ROOT / 'docs/modernization/runtime/baseline'
    destination.mkdir(parents=True, exist_ok=True)
    original = Path.cwd()
    with tempfile.TemporaryDirectory(dir=ROOT.parent / '.runtime-tests') as tmp:
        try:
            os.chdir(tmp)
            atomic_fixture(Path('raw_data'))
            config = dict(dataset='Tiny', data_col=['speed'], weight_col='distance',
                          input_window=4, output_window=2, batch_size=2, num_workers=0,
                          train_rate=.6, eval_rate=.2, cache_dataset=False, scaler='none',
                          pad_with_last_sample=False, rnn_type='GRU', hidden_size=4,
                          num_layers=1, dropout=0., teacher_forcing_ratio=0., device=torch.device('cpu'))
            dataset = TrafficStatePointDataset(config)
            train, _, _ = dataset.get_data()
            batch = {'X': torch.tensor(np.stack([item[0] for item in train.dataset[:2]]), dtype=torch.float32),
                     'y': torch.tensor(np.stack([item[1] for item in train.dataset[:2]]), dtype=torch.float32)}
            torch.manual_seed(17)
            torch.set_num_threads(1)
            model = RNN(config, dataset.get_data_feature()).eval()
            with torch.no_grad():
                prediction = model.predict(batch)
                value = masked_mae_torch(prediction, batch['y'].clone(), 0).item()
            result = dict(source_commit='6ddc6ee7d0dc8467831c678e69b9e70aa131a763', seed=17,
                          config={key: val for key, val in config.items() if key != 'device'},
                          X=batch['X'].tolist(), y=batch['y'].tolist(), prediction=prediction.tolist(),
                          masked_mae=value, weights={key: val.tolist() for key, val in model.state_dict().items()})
        finally:
            os.chdir(original)
    (destination / 'gru.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    suite = ET.parse(ROOT.parent / '.runtime-tests/architecture-baseline.xml').getroot().find('testsuite')
    environment = dict(source_commit=result['source_commit'], python=platform.python_version(),
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
        cuda=torch.cuda.is_available(), tests=int(suite.attrib['tests']), failures=int(suite.attrib['failures']),
        errors=int(suite.attrib['errors']), skipped=int(suite.attrib['skipped']), seconds=float(suite.attrib['time']),
        install='python -m pip install -e .[dev]', test_command='python -m pytest -q',
        dependencies=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines())
    assert environment['tests'] == 131 and environment['failures'] == environment['errors'] == 0
    (destination / 'environment.json').write_text(json.dumps(environment, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: environment[key] for key in ('python','torch','tests','failures','seconds')}))


if __name__ == '__main__':
    main()
