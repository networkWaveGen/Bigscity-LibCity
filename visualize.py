"""Convert atomic data to GeoJSON using explicit data/output directories."""
import argparse
from libcity.utils.visualize import VisHelper


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    VisHelper(vars(args)).visualize()


if __name__=='__main__':
    main()
