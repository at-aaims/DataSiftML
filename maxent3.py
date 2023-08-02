"""This script computes MaxEnt using Murali's approach with vorticity"""

import dataloader
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from args import args
from sklearn.cluster import KMeans

figsize = (10, 2)

dl = dataloader.DataLoader(args.path)

write_interval = 100
num_time_steps = 100

x, y = dl.load_xyz()

_, vorticity = dl.load_multiple_timesteps(write_interval, num_time_steps)
print(vorticity.shape)

#for timestep in range(vorticity.shape[0]):
if True:
    timestep = 70

    # K-means clustering
    data = vorticity[timestep, :].reshape(-1, 1)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
    kmeans.fit(data)
    y_pred = kmeans.predict(data)
    print(y_pred)
    print(y_pred.shape)

    if args.plot:
        plt.figure(figsize=figsize)
        plt.scatter(x, y, c=kmeans.labels_, cmap='tab10')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('KMeans clustering of vorticity')
        plt.colorbar()
        plt.show()

    clusters = [data[np.argwhere(y_pred == i).flatten()] for i in range(args.n_clusters)]
    clusters = [cluster.flatten() for cluster in clusters]
    print(clusters)

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
            adj_matrix[i, j] = scipy.stats.entropy(p, q)

    df = pd.DataFrame(adj_matrix)
    print(df)

    # Create a graph from the adjacency matrix and compute the minimum cut
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())

    # Select top clusters according to cutoff_threshold
    cutoff_threshold = 0.5
    in_strengths = np.sum(adj_matrix, axis=0)
    out_strengths = np.sum(adj_matrix, axis=1)
    # for verification
    #in_strengths = dict(G.in_degree(weight='weight')).values()
    #out_strengths = dict(G.out_degree(weight='weight')).values()
    print("in-strengths:", in_strengths)
    print("out-strengths:", out_strengths)

    top_instrength = np.argsort(in_strengths)
    print("sorted in_strength:", in_strengths[top_instrength])
    top_outstrength = np.argsort(out_strengths) 

    # control vs effect - perturbation vs effect
    print(top_instrength)  # id's of clusters sorted based on in_strength
    print(top_outstrength)

    threshold = cutoff_threshold * np.sum(in_strengths) # same as when computing with out_strength
    print(threshold)

    sumstrength = 0
    i = len(in_strengths) - 1
    optimal_subset = []
   
    while sumstrength < threshold:
        optimal_subset.append(top_instrength[i])
        sumstrength += in_strengths[top_instrength[i]]
        i -= 1

    print('optimal subset of clusters:', optimal_subset)

    num_samples = len(kmeans.labels_)
    mask = np.isin(kmeans.labels_, optimal_subset)
    num_samples_compressed = len(kmeans.labels_[mask])
    print(f"uncompressed samples: {num_samples}, filtered subset: {num_samples_compressed},", 
          f"compression factor: {num_samples / num_samples_compressed:.1f}X")

    # Find the indices of the original dataset, data, that have optimal clusters
    subsampled_data = data[mask].ravel()
    print(subsampled_data)

    # Show only optimal clusters
    if args.plot:
        # set clusters below threshold to -1 
        kmeans.labels_[~mask] = -1 
        # and set their color to white so they won't be visible
        cmap_white_first = mcolors.ListedColormap(['white', *plt.cm.viridis.colors])
        plt.figure(figsize=figsize)
        plt.scatter(x, y, c=kmeans.labels_, cmap=cmap_white_first, vmin=-0.5, vmax=max(kmeans.labels_) + 0.5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Features with highest entropy')
        cbar = plt.colorbar(ticks=np.arange(0, max(kmeans.labels_), 1))
        cbar.set_label('Cluster Label')
        plt.show()

