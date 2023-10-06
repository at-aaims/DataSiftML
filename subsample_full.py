""" train on entire dataset - this file loads the data and saves to subsampled.npz """

import dataloader
import numpy as np
import os

from args import args
from constants import *

dfpath = os.path.join(SNPDIR, DRAWFN)

if args.window > 1: 
    raise ValueError("windowing not yet supported in this version")

dl = dataloader.DataLoader(args.path)
x, y = dl.load_xyz()
X, Y = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.target)
print(X.shape, Y.shape, args.num_timesteps)
if args.cluster_var == args.target:
    cv = Y
else:
    _, cv = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.cluster_var)

outfile = os.path.join(SNPDIR, 'subsampled.npz')
np.savez(outfile, X=X, Y=Y, cv=cv, x=x, y=y, target=args.target)
print(f"output file {outfile}")

print(X.shape, Y.shape)
