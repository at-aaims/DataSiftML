"""This script computes MaxEnt using Murali's approach with a collective variable
   such as vorticity, pressure, or a combination"""

import dataloader
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from args import args
from itertools import cycle
from sklearn.cluster import KMeans

figsize = (10, 2)

dl = dataloader.DataLoader(args.path)

x, y = dl.load_xyz()

X, Y = dl.load_multiple_timesteps(args.write_interval, args.num_time_steps, target=args.target)
print(X.shape, Y.shape)

#if True:
#    timestep = 70

for timestep in range(Y.shape[0]):

    # K-means clustering
    data = Y[timestep, :].reshape(-1, 1)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    y_pred = kmeans.predict(data)
    #print(y_pred)
    print(y_pred.shape)

    if args.plot:
        plt.figure(figsize=figsize)
        plt.scatter(x, y, c=kmeans.labels_, marker='.', cmap='tab10')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'KMeans clustering of {args.target}')
        plt.colorbar()
        plt.show()

    # Following is from Greg
    if False:
        # Use all colors that matplotlib provides by default.
        colors_ = cycle(colors.cnames.keys())

        fig, ax = plt.subplots(figsize=figsize)
        alpha = 1.0
        for this_centroid, k, col in zip(centroids, range(args.n_clusters), colors_):
            print(this_centroid, k, col)
            mask = labels == k
            ax.scatter(x[mask], y[mask], c=col, marker='.', alpha=alpha)

        ax.set_autoscaley_on(False)
        title = 'num_clusters = %s ' % str(args.n_clusters)
        ax.set_title( title )

        plt.show()

    clusters = [data[np.argwhere(y_pred == i).flatten()] for i in range(args.n_clusters)]
    clusters = [cluster.flatten() for cluster in clusters]
    #print(clusters)

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

    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
    
    total_entropy = np.sum(adj_matrix)
    print(f"total entropy: {total_entropy}")

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
    subsampled_X = X[timestep, mask, :]
    print(subsampled_X.shape)
    # (100, 10800, 2) (100, 10800)
    subsampled_Y = np.expand_dims(data[mask].ravel(), axis=1)
    print(subsampled_Y.shape)

    df = pd.DataFrame(np.concatenate((subsampled_Y, subsampled_X), axis=1), columns=[args.target, 'u', 'v'])
    df.to_csv(f"data_{timestep:05}.csv", index=False)

    # Show only optimal clusters
    #if args.plot:
    if True:
        # set clusters below threshold to -1 
        kmeans.labels_[~mask] = -1 
        # and set their color to white so they won't be visible
        cmap_white_first = colors.ListedColormap(['white', *plt.cm.viridis.colors])
        plt.figure(figsize=figsize)
        plt.scatter(x, y, c=kmeans.labels_, marker='.', cmap=cmap_white_first, \
                    vmin=-0.5, vmax=max(kmeans.labels_) + 0.5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Features with highest entropy')
        cbar = plt.colorbar(ticks=np.arange(0, max(kmeans.labels_), 1))
        cbar.set_label('Cluster Label')
        #plt.show()
        plt.savefig(f'frame_{timestep:04d}.png', dpi=100)
