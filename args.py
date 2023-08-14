import argparse
import os

parser = argparse.ArgumentParser()
archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
parser.add_argument('--arch', type=str, default='fcn', choices=archs, help='Type of neural network architecture')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('-cv', '--cluster_var', type=str, default='p', choices=['drag', 'p', 'wz', 'pwz', 'stream'], help='cluster variable')
parser.add_argument('--cutoff', type=float, default=0.5, help='optimal data cutoff factor, e.g., 0.1 keep top ten percent')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('-nc', '--num_clusters', type=int, default=10, help='number of clusters')
parser.add_argument('-ns', '--num_samples', type=int, default=100, help='number of subsamples')
parser.add_argument('--num_time_steps', type=int, default=100, help='OpenFOAM number of timestamps')
parser.add_argument('--path', type=str, default='.', help='path to simulation')
parser.add_argument('--plot', action='store_true', default=False, help='show plots')
choices = ['random', 'random-weighted', 'lhs', 'maxent', 'none', 'silhouette']
parser.add_argument('--subsample', type=str, default='none', choices=choices, help='sampling strategy')
choices = ['StandardScaler', 'MinMaxScaler', 'PowerTransformer']
parser.add_argument('--scaler', type=str, default='StandardScaler', choices=choices, help='scaler function')
parser.add_argument('--test_frac', type=float, default=0.1, help='fraction of data to hold out for testing')
parser.add_argument('--target', type=str, default='wz', choices=['drag', 'p', 'wz', 'pwz', 'stream'], help='training target')
parser.add_argument('--local', action='store_true', default=False, help='local (as opposed to global) field prediction')
parser.add_argument('--time', type=str, default='1000', help='time step to analyze')
parser.add_argument('--sequence', type=bool, default=False, help='Aggregate individual time-steps into a sequence')
parser.add_argument('--tune', action='store_true', default=False, help='run hyperparameter optimization')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose output')
parser.add_argument('--window', type=int, default=1, help='time window sequence size')
parser.add_argument('--write_interval', type=int, default=100, help='OpenFOAM write interval')
args = parser.parse_args()
