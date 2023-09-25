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

num_timesteps = X.shape[0] // args.window * args.window
print('num_timesteps:', X.shape[0])

Xout = np.zeros((num_timesteps, args.num_samples, 2))

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

        #if args.field_prediction_type == FPT_GLOBAL:
        #    subsampled_X = X[indices, :]
        #else:
        #    subsampled_X, subsampled_Y = X[indices, :], Y[indices]

        if args.verbose: print(subsampled_X.shape, subsampled_Y.shape)

        Xout[ts, :, :] = subsampled_X
        Yout[ts, :] = subsampled_Y

        if args.plot:
            plt.clf()
            plt.figure(figsize=(10, 2))
            plt.scatter(x[indices], y[indices], marker='.', vmin=-0.5, vmax=0.5)
            plt.xlim([-20, 60])
            plt.ylim([-10, 10])
            plt.savefig(os.path.join(PLTDIR, f'frame_{ts:04d}_{args.subsample}.png'), dpi=100)

        ts += 1


print(Xout.shape, Yout.shape)
np.savez(os.path.join(SNPDIR, 'subsampled.npz'), X=Xout, Y=Yout, target=args.target)