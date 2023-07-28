"""This script computes MaxEnt using Murali's approach with vorticity"""

import dataloader
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from args import args
from sklearn.cluster import KMeans

dl = dataloader.DataLoader(args.path)

write_interval = 100
num_time_steps = 100

x, y = dl.load_xyz()

_, vorticity = dl.load_multiple_timesteps(write_interval, num_time_steps)
print(vorticity.shape)

for timestep in range(vorticity.shape[0]):

    # K-means clustering
    data = vorticity[timestep, :].reshape(-1, 1)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
    kmeans.fit(data)
    y_pred = kmeans.predict(data)
    print(y_pred.shape)

    if args.plot:
        plt.scatter(x, y, c=kmeans.labels_, cmap='viridis')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('KMeans Clustering')
        plt.show()

    clusters = [data[np.argwhere(y_pred == i).flatten()] for i in range(args.n_clusters)]
    clusters = [cluster.flatten() for cluster in clusters]

    # Initialize a list to store your probability distributions and their bin edges
    prob_dists = []
    bin_edges_list = []

    # Specify a consistent bin range and count
    bin_range = (np.min([np.min(cluster) for cluster in clusters]), 
                 np.max([np.max(cluster) for cluster in clusters]))
    num_bins = 50  # or choose another suitable value

    for cluster in clusters:
        counts, bin_edges = np.histogram(cluster, bins=num_bins, range=bin_range, density=False)
        prob_dist = counts / np.sum(counts)
        prob_dists.append(prob_dist)
        bin_edges_list.append(bin_edges)

    if args.plot:
        # Create a grid of probability distribution subplots
        grid_size = math.ceil(math.sqrt(len(clusters)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(9, 9))
        axes = axes.flatten()
        for i, (prob_dist, bin_edges, ax) in enumerate(zip(prob_dists, bin_edges_list, axes)):
            ax.bar(bin_edges[:-1], prob_dist, width=np.diff(bin_edges), align="edge")
            ax.set_title(f'Cluster {i + 1}')
        for ax in axes[len(clusters):]: ax.remove()
        plt.tight_layout()
        plt.show()

    n_dists = args.n_clusters

    # Compute adjacency matrix containing relative entropy for each pair of distributions
    adj_matrix = np.zeros((n_dists, n_dists))

    for i in range(n_dists):
        for j in range(n_dists):
            p = prob_dists[i] + 1e-10 # to avoid division by zero
            q = prob_dists[j] + 1e-10 # to avoid division by zero
            adj_matrix[i, j] = np.sum(scipy.stats.entropy(p, q))

    df = pd.DataFrame(adj_matrix)
    print(df)

    # Create a graph from the adjacency matrix and compute the minimum cut
    G = nx.from_numpy_array(adj_matrix)

    source = 0
    sink = 1

    # Compute the minimum cut
    #cut_value, partition = nx.minimum_cut(G, source, sink)
    cut_value, partition = nx.stoer_wagner(G)

    print("Cut value:", cut_value)
    print("Partition:", partition)

