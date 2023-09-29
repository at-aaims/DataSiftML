import dataloader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np 
import os

from args import args
from constants import *


def subsample_random(X, num_samples, random_seed=[0]):
    random_seed[0] += 1
    print('random seed: ', random_seed[0])
    np.random.seed(random_seed[0])
    return np.random.choice(X.shape[1], num_samples, replace=False)


dfpath = os.path.join(SNPDIR, DRAWFN)

if os.path.exists(dfpath):
    data = np.load(dfpath)
    X, Y, cv, x, y = data['X'], data['Y'], data['cv'], data['x'], data['y']

else:
    dl = dataloader.DataLoader(args.path)
    x, y = dl.load_xyz()
    X, Y = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.target)
    print(X.shape, Y.shape, args.num_timesteps)
    if args.cluster_var == args.target:
        cv = Y
    else:
        _, cv = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.cluster_var)

    np.savez(dfpath, X=X, Y=Y, cv=cv, x=x, y=y)
    print(f"output file {dfpath}")

if args.dtype == "structured":

    dfpath = os.path.join(SNPDIR, 'interpolated.npz')
    data = np.load(dfpath)
    x, y, X, _, cv = data['x'], data['y'], data['X'], data['Y'], data['cv']

    x, y = x[0], y[0] # grid points should not change over time
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    X = X.reshape(X.shape[0], -1, X.shape[-1])[1:]
    cv = cv.reshape(cv.shape[0], -1)[1:]

print(x.shape, y.shape, X.shape, Y.shape, cv.shape)

num_timesteps = X.shape[0] // args.window * args.window
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
        Yout[ts, :] = subsampled_Y

        if args.plot:
            plt.clf()
            #plt.rcParams.update({'font.size': 15})
            plt.figure(figsize=(9, 2), facecolor='1')
            plt.scatter(x[indices], y[indices], marker='.', vmin=-0.5, vmax=0.5)
            plt.axis('equal')
            plt.xlim([-25, 65])
            plt.ylim([-10, 10])
            plt.savefig(os.path.join(PLTDIR, f'frame_{ts:04d}_random.png'), dpi=100)

            plt.clf()
            colors_ = plt.cm.get_cmap('tab10', args.num_clusters)

        ts += 1


print(Xout.shape, Yout.shape)
outfile = os.path.join(SNPDIR, 'subsampled.npz')
arrays = { 'X': Xout, 'Y': Yout, 'x': x[indices], 'y': y[indices], 'target': args.target }
np.savez(outfile, **arrays)
if args.subsample != "proportional": print('min number of samples over all timesteps:', mins)
print(f'output {outfile}')
