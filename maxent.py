"""This script computes MaxEnt directly on primitive flow variables 
   using kernel density estimation (KDE)"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluidfoam import readscalar, readvector, readforce
from pyDOE import lhs
from sklearn.cluster import KMeans, Birch
from sklearn.neighbors import KernelDensity

from args import args

def calculate_entropy(data):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
    log_dens = kde.score_samples(data)
    entropy = -np.sum(np.exp(log_dens) * log_dens)
    return entropy

# Read solution values from OpenFOAM simulation
p = readscalar(args.path, args.time, 'p.gz')
x, y, z = readvector(args.path, args.time, 'C.gz')
Ux, Uy, Uz = readvector(args.path, args.time, 'U.gz')
wx, wy, wz = readvector(args.path, args.time, 'vorticity.gz')
forces = readforce(args.path, time_name='0', name='forces')

# Drag force is composed of both a viscous and pressure components
drag = forces[:, 1] + forces[:, 2]

if args.verbose:
    print('force.shape:', forces.shape)
    print('drag.shape:', drag.shape)
    print(drag)

#if args.plot:
#    plt.figure()
#    plt.plot(forces[:, 0], drag)
#    plt.xlabel('t')
#    plt.ylabel('drag')
#    plt.ylim(-10, 10) 
#    plt.title('Drag history')
#    plt.show()

# Add an extra dimension
p = np.expand_dims(p, axis=1)
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)
Ux = np.expand_dims(Ux, axis=1)
Uy = np.expand_dims(Uy, axis=1)
wz = np.expand_dims(wz, axis=1)

stacked = np.hstack((x, y, p, Ux, Uy, wz))

if args.verbose:
    print(p.shape)
    print(x.shape)
    print(y.shape)
    print(Ux.shape)
    print(Uy.shape)
    print(wz.shape)
    print(stacked.shape)

df = pd.DataFrame(stacked, columns=['x', 'y', 'p', 'Ux', 'Uy', 'wz'])
#df_sub = df[['p', 'Ux', 'Uy']]
df_sub = df[['p', 'wz']]

# Output CSV file named by timestamp, e.g. 1000.csv
file_name = args.time + '.csv'
df.to_csv(file_name, index=False)

with open(file_name, 'a') as file:
    drag_value = drag[int(args.time)]
    file.write(f"# Drag: {drag_value}\n")

# K-means clustering
kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
kmeans.fit(df_sub)
df_sub['cluster'] = kmeans.predict(df_sub)

# Plot
if args.plot:
    plt.scatter(x, y, c=kmeans.labels_, cmap='viridis')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('KMeans Clustering')
    plt.show()

# Print the cluster centers
if args.verbose:
    print("Cluster centers:")
    print(kmeans.cluster_centers_)

# Perform sampling of clusters
n_samples_per_cluster = args.n_subsamples // args.n_clusters
print(f'num samples per cluster: {n_samples_per_cluster}')
samples = []
total_points = len(df_sub)
print(f"total points: {total_points}")
for cluster in range(args.n_clusters):
    cluster_points = df_sub[df_sub['cluster'] == cluster]
    # Proportional sampling according to cluster size
    n_samples = int(len(cluster_points) / total_points * n_samples_per_cluster)
    cluster_samples = cluster_points.sample(n_samples)
    samples.append(cluster_samples)

# Concatenate all samples into a single DataFrame
samples_df = pd.concat(samples)

# Calculate entropy of original data
original_entropy = calculate_entropy(df_sub.values)
print(f'original entropy: {original_entropy}')

# Calculate entropy of sampled data
sampled_entropy = calculate_entropy(samples_df.values)
print(f'cluster-sampled entropy: {sampled_entropy}')

# LHS
n_dims = df_sub.shape[1]
lhs_sample = lhs(n_dims, samples=args.n_subsamples)
for i in range(n_dims):
    lhs_sample[:, i] = lhs_sample[:, i] * (df_sub.iloc[:, i].max() - df_sub.iloc[:, i].min()) + df_sub.iloc[:, i].min()

# Calculate entropy of LHS 
sampled_entropy = calculate_entropy(lhs_sample)
print(f'LHS-sampled entropy: {sampled_entropy}')
