import argparse
import glob
import numpy as np

from pydmd import DMD
import matplotlib.pyplot as plt

from fluidfoam import readscalar, readvector, readforce

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='.', help="path to simulation")
args = parser.parse_args()

nx = 10800
nt = 100

p = np.zeros((nt, nx))
u = np.zeros((nt, nx))
v = np.zeros((nt, nx))
wz = np.zeros((nt, nx))

# Read solution values from OpenFOAM simulation
for i, ts in enumerate(range(100, 10100, 100)):
    print(i, ts)
    p[i, :] = readscalar(args.path, str(ts), 'p.gz')
    u[i, :], v[i, :], _ = readvector(args.path, str(ts), 'U.gz')
    _, _, wz[i, :] = readvector(args.path, str(ts), 'vorticity.gz')

# Concatenate the data along the second dimension
snapshots = np.concatenate([p, u, v, wz], axis=1)

# Apply DMD
dmd = DMD(svd_rank=2)
dmd.fit(snapshots.T)

# Print eigenvalues and plot modes and dynamics
for idx, mode in enumerate(dmd.modes.T):
    plt.figure(figsize=(16,4))
    plt.subplot(121, title='Mode {}'.format(idx+1))
    plt.plot(mode.real)
    plt.plot(mode.imag)
    plt.subplot(122, title='Dynamics {}'.format(idx+1))
    for dynamic in dmd.dynamics:
        plt.plot(dynamic.real)
        plt.plot(dynamic.imag)
plt.show()
