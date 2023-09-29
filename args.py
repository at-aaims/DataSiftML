import argparse
import os
import yaml

from constants import FPT_LOCAL, FPT_GLOBAL

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
parser.add_argument('--arch', type=str, default='fcn', choices=archs, help='Type of neural network architecture')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('-cv', '--cluster_var', type=str, default='p', choices=['drag', 'p', 'wz', 'pwz', 'stream'], help='cluster variable')
parser.add_argument('--cutoff', type=float, default=0.5, help='optimal data cutoff factor, e.g., 0.1 keep top ten percent')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--hybrid', type=float, default=1, help='hybrid maxent+random sampling approach')
parser.add_argument('-nc', '--num_clusters', type=int, default=10, help='number of clusters')
parser.add_argument('-ns', '--num_samples', type=int, default=100, help='number of subsamples')
parser.add_argument('--num_timesteps', type=int, default=100, help='OpenFOAM number of timestamps')
parser.add_argument('--path', type=str, default='.', help='path to simulation')
parser.add_argument('--patience', type=int, default=5, help='number epochs for early stopping')
parser.add_argument('--plot', action='store_true', default=False, help='show plots')
choices = ['random', 'random-weighted', 'silhouette', 'proportional', 'equal', 'equalpercentage']
parser.add_argument('--subsample', type=str, default='proportional', choices=choices, help='sampling strategy')
choices = ['StandardScaler', 'MinMaxScaler', 'PowerTransformer', 'GaussRankScaler']
parser.add_argument('--xscaler', type=str, default='MinMaxScaler', choices=choices, help='scaler function')
parser.add_argument('--yscaler', type=str, default='GaussRankScaler', choices=choices, help='scaler function')
parser.add_argument('--dtype', type=str, default='unstructured', choices=['structured', 'unstructured'], help='data type')
parser.add_argument('--test_frac', type=float, default=0.1, help='fraction of data to hold out for testing')
parser.add_argument('--target', type=str, default='wz', choices=['drag', 'p', 'wz', 'pwz', 'stream'], help='training target')
parser.add_argument('--time', type=str, default='1000', help='time step to analyze')
parser.add_argument('--sequence', action='store_true', default=False, help='aggregate individual time-steps into a sequence')
parser.add_argument('--tune', action='store_true', default=False, help='run hyperparameter optimization')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='verbose output')
parser.add_argument('--window', type=int, default=1, help='time window sequence size')
parser.add_argument('--write_interval', type=int, default=100, help='OpenFOAM write interval')
args = parser.parse_args()

args.field_prediction_type = FPT_GLOBAL if args.target == 'drag' else FPT_LOCAL
if args.arch == 'lstm': args.sequence = True

fn = './defaults.yaml'

if os.path.exists(fn):
    with open(fn, 'r') as yaml_file:
        defaults = yaml.safe_load(yaml_file)
    for key, value in defaults.items():
        setattr(args, key, value)

print(args)
