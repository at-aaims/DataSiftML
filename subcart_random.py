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

dfpath2 = os.path.join('data/npzdata', 'cyl_data400200.npz')

dl = dataloader.DataLoader(args.path)
Xold, Y = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.target)
print('Y shape')
print(Y.shape)
np.savez(dfpath, Y=Y)

#if os.path.exists(dfpath):
#    data = np.load(dfpath)
#    X, Y, cv, x, y = data['X'], data['Y'], data['cv'], data['x'], data['y']

if os.path.exists(dfpath2):
    print('Reading custom data')
    #unsdata = np.load(dfpath)
    #Y = unsdata['Y']
    #print('drag',Y.shape)
    data = np.load(dfpath2)
    print(list(data.keys()))
    x, y, X, Yp, Wz = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'] #, data['y']

#else:
#    dl = dataloader.DataLoader(args.path)
#    x, y = dl.load_xyz()
#    X, Y = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.target)
#    print(X.shape, Y.shape, args.num_timesteps)
#    if args.cluster_var == args.target:
#        cv = Y
#    else:
#        _, cv = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.cluster_var)
#
#    np.savez(dfpath, X=X, Y=Y, cv=cv, x=x, y=y)
#    print(f"output file {dfpath}")

# Over-ride standalone subsample_random by reading custom data
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2],X.shape[-1])
X = X[1:]
x = x[1:]
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[-1])
print(x.shape)
y = y[1:]
y = y.reshape(y.shape[0],y.shape[1]*y.shape[2],y.shape[-1])
print(y.shape)

num_timesteps = X.shape[0] // args.window * args.window
print('num_timesteps:', X.shape[0])

Xout = np.zeros((num_timesteps, args.num_samples, 2))

if args.field_prediction_type == FPT_GLOBAL: # global quantity prediction
    Yout = np.zeros((num_timesteps, 1))
else: # local field prediction
    Yout = np.zeros((num_timesteps, args.num_samples))

for timestep in range(0, num_timesteps - args.window, args.window):

    print(f"\nTIMESTEP: {timestep}-{timestep + args.window}\n")

    indices = subsample_random(x, args.num_samples)

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
            plt.figure(figsize=(9, 2))
            plt.scatter(x[0][indices], y[0][indices], marker='.', vmin=-0.5, vmax=0.5)
            plt.xlim([-25, 65])
            plt.ylim([-10, 10])
            plt.axis('equal')
            plt.savefig(os.path.join(PLTDIR, f'frame_{ts:04d}_{args.subsample}.png'), dpi=100)

        ts += 1


print(Xout.shape, Yout.shape)
np.savez(os.path.join(SNPDIR, 'subsampled.npz'), X=Xout, Y=Yout, target=args.target)
