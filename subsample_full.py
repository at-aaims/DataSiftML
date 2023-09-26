""" train on entire dataset - this file loads the data and saves to subsampled.npz """

import dataloader
import numpy as np
import os

from args import args
from constants import *

dfpath = os.path.join(SNPDIR, DRAWFN)

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

print(X.shape, Y.shape)
