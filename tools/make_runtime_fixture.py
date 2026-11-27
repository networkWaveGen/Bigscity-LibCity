"""Create a reproducible small source-runtime example outside the source tree."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from runtime_fixtures import atomic_fixture,map_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--task',choices=('traffic_state_pred','map_matching'),default='traffic_state_pred')
    options = parser.parse_args()
    directory = options.directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if options.task=='map_matching':
        map_fixture(directory/'data')
        config=dict(data=dict(dataset='RoadTiny',data_dir='data',cache_dir='cache'),model={'name':'HMMM'},
                    runtime=dict(device='cpu',seed=17,output_dir='runs'))
        filename='hmmm.json'
    else:
        atomic_fixture(directory / 'data')
        config = dict(data=dict(dataset='Tiny', data_dir='data', cache_dir='cache', input_window=4, output_window=2),
            model=dict(name='GRU', params={'hidden_size':4}), training=dict(epochs=2, batch_size=8),
            runtime=dict(device='cpu', seed=17, output_dir='runs'))
        filename='gru.json'
    path = directory / filename
    path.write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
