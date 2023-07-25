import argparse
import os

parser = argparse.ArgumentParser()
archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
parser.add_argument("-nc", "--n_clusters", type=int, default=10, help="number of clusters")
parser.add_argument("-ns", "--n_subsamples", type=int, default=100, help="number of subsamples")
parser.add_argument('--arch', type=str, default='lstm', choices=archs, help='Type of neural network architectures')
parser.add_argument("--path", type=str, default='.', help="path to simulation")
parser.add_argument("--time", type=str, default='1000', help="time step to analyze")
parser.add_argument("--plot", action='store_true', default=False, help="show plots")
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--batch", type=int, default=32, help="batch size")
parser.add_argument("--verbose", action='store_true', default=False, help="verbose output")
args = parser.parse_args()


