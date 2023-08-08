import dataloader
import numpy as np

from args import args
from pyDOE import lhs

dl = dataloader.DataLoader(args.path)

x, y = dl.load_xyz()
X, Y = dl.load_multiple_timesteps(args.write_interval, args.num_time_steps, target=args.target)

print(X.shape, Y.shape)

if args.sampling == "random":
    indices = np.random.choice(X.shape[0], args.num_samples, replace=False)
elif args.sampling == "lhs":
    indices = lhs(X.shape[0], args.num_samples)
else:
    raiseValueError(f"sampling strategy {args.sampling} not supported")

print(indices)

X_subsampled, Y_subsampled = X[:,indices,:], Y[:,indices]
print(X_subsampled.shape, Y_subsampled.shape)
