"""Convert old LibCity checkpoints to CPU tensor/state-dict formats."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libcity.utils.checkpoint import migrate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--trust-legacy-pickle', action='store_true',
                        help='Allow object unpickling only for a checkpoint whose source you trust.')
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error('Use a separate destination so the original checkpoint is preserved.')
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    migrate_checkpoint(args.source, args.destination, trusted_legacy=args.trust_legacy_pickle)
    print(f'Wrote CPU state dictionaries to {args.destination}')


if __name__ == '__main__':
    main()
