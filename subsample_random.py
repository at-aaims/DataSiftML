import dataloader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np 
import os

from args import args
from constants import *
from helpers import load


def subsample_random(X, num_samples, random_seed=[0]):
    random_seed[0] += 1
    if not args.noseed:
        print('random seed: ', random_seed[0])
        np.random.seed(random_seed[0])
    return np.random.choice(X.shape[1], num_samples, replace=False)


dfpath = os.path.join(SNPDIR, DRAWFN)

if os.path.exists(dfpath):
    data = load(dfpath)
    X, Y, cv, x, y, z = data['X'], data['Y'], data['cv'], data['x'], data['y'], data['z']

else:
    if args.dtype == "csv":
        dl = dataloader.DataLoaderCSV(args.path, dims=args.dims)
    else:
        dl = dataloader.DataLoaderOF(args.path, dims=args.dims)
    x, y, z = dl.load_xyz()
    X, Y, cv = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, \
                                          target=args.target, cv=args.cluster_var)
    print(X.shape, Y.shape, args.num_timesteps)

    np.savez(dfpath, X=X, Y=Y, cv=cv, x=x, y=y, z=z)
    print(f"output file {dfpath}")

if args.dtype == "interpolated":

    dfpath = os.path.join(SNPDIR, 'interpolated.npz')
    data = np.load(dfpath)
    x, y, X, _, cv = data['x'], data['y'], data['X'], data['Y'], data['cv']

    x, y = x[0], y[0] # grid points should not change over time
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    X = X.reshape(X.shape[0], -1, X.shape[-1])[1:]
    cv = cv.reshape(cv.shape[0], -1)[1:]

print(x.shape, y.shape, X.shape, Y.shape, cv.shape)

num_timesteps = X.shape[0] // args.window * args.window + 1
print('num_timesteps:', X.shape[0])

Xout = np.zeros((num_timesteps, args.num_samples, X.shape[2]))

if args.field_prediction_type == FPT_GLOBAL: # global quantity prediction
    Yout = np.zeros((num_timesteps, 1))
else: # local field prediction
    Yout = np.zeros((num_timesteps, args.num_samples))

for timestep in range(0, num_timesteps - args.window, args.window):

    print(f"\nTIMESTEP: {timestep}-{timestep + args.window}\n")

    indices = subsample_random(X, args.num_samples)

    ts = timestep 
    for sub_timestep in range(args.window):
        if args.verbose: print(f"timestep: {ts}")

        # Find the indices of the original dataset, data, that have optimal clusters
        print(ts, len(indices), X.shape, X[ts, indices].shape)
        subsampled_X = X[ts, indices, :]
        subsampled_Y = Y[ts] if args.field_prediction_type == FPT_GLOBAL else Y[ts, indices]

        if args.verbose: print(subsampled_X.shape, subsampled_Y.shape)

        Xout[ts, :, :] = subsampled_X
        try:
            Yout[ts, :] = subsampled_Y
        except Exception as e:
            raise Exception("Try removing ./snapshots/raw_data.npz and re-running" + str())

    if args.plot:

        plt.clf()
        if args.dims == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.subplot(111, projection='3d')
            ax.view_init(elev=20., azim=-35)
            ax.scatter(x[indices], y[indices], z[indices], marker='.', vmin=-0.5, vmax=0.5)
        else:
            plt.figure(figsize=(9, 2))
            plt.scatter(x[indices], y[indices], marker='.', vmin=-0.5, vmax=0.5)
            plt.xlim([-25, 65])
            plt.ylim([-10, 10])
        plt.axis('equal')
        plt.savefig(os.path.join(PLTDIR, f'frame_{ts:04d}_random.png'), dpi=100, bbox_inches='tight')

    ts += 1


print(Xout.shape, Yout.shape)
outfile = os.path.join(SNPDIR, 'subsampled.npz')
arrays = { 'X': Xout, 'Y': Yout, 'x': x[indices], 'y': y[indices], 'target': args.target }
np.savez(outfile, **arrays)
if args.subsample != "proportional": print('min number of samples over all timesteps:', mins)
print(f'output {outfile}')
