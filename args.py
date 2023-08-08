import argparse
import os

parser = argparse.ArgumentParser()
archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
parser.add_argument("-nc", "--n_clusters", type=int, default=10, help="number of clusters")
parser.add_argument("-ns", "--n_subsamples", type=int, default=100, help="number of subsamples")
parser.add_argument('--arch', type=str, default='lstm', choices=archs, help='Type of neural network architectures')
parser.add_argument("--path", type=str, default='.', help="path to simulation")
parser.add_argument("--test_frac", type=float, default=0.1, help="fraction of data to hold out for testing")
parser.add_argument("--target", type=str, default='wz', choices=['p', 'wz', 'pwz', 'stream'], help="training target")
parser.add_argument("--time", type=str, default='1000', help="time step to analyze")
parser.add_argument("--tune", action='store_true', default=False, help="run hyperparameter optimization")
parser.add_argument("--plot", action='store_true', default=False, help="show plots")
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
choices = ['StandardScaler', 'MinMaxScaler', 'PowerTransformer']
parser.add_argument("--scaler", type=str, default="StandardScaler", choices=choices, help="scaler function")
parser.add_argument("--batch", type=int, default=32, help="batch size")
parser.add_argument("--cutoff", type=float, default=0.5, help="optimal data cutoff factor, e.g., 0.1 keep top ten percent")
parser.add_argument("--verbose", action='store_true', default=False, help="verbose output")
parser.add_argument("--num_time_steps", type=int, default=100, help="OpenFOAM number of timestamps")
parser.add_argument("--write_interval", type=int, default=100, help="OpenFOAM write interval")
args = parser.parse_args()

