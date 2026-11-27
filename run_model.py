"""Source entrypoint; experiment behavior lives in libcity.pipeline.experiment."""
import argparse
import json

from libcity.pipeline.experiment import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run a LibCity experiment')
    parser.add_argument('mode', choices=('train', 'evaluate'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--set', action='append', default=[], metavar='BLOCK.KEY=JSON')
    arguments = parser.parse_args(argv)
    overrides = {}
    for item in arguments.set:
        if '=' not in item:
            parser.error('--set requires BLOCK.KEY=JSON')
        key, value = item.split('=', 1)
        keys = key.split('.')
        if len(keys) < 2 or any(not part for part in keys):
            parser.error('--set requires a dotted configuration key')
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        block = overrides
        for part in keys[:-1]:
            block = block.setdefault(part, {})
        block[keys[-1]] = value
    result = run_experiment(arguments.mode, arguments.config, overrides=overrides,
                            checkpoint=arguments.checkpoint, resume=arguments.resume)
    print(json.dumps(dict(metrics=result.metrics, artifacts={key: str(path) for key, path in result.artifacts.items()})))
    return result


if __name__ == '__main__':
    main()
