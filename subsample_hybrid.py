"""This script computes MaxEnt using Murali's approach with a collective variable
   such as vorticity, pressure, or a combination"""

import dataloader
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx
import numpy as np
import os
import pandas as pd
import scipy

from args import args
from constants import *
from helpers import scale_probabilities
from itertools import cycle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


figsize = (10, 2)

dfpath = os.path.join(SNPDIR, DRAWFN)

if os.path.exists(dfpath):
    data = np.load(dfpath)
    X, Y, cv, x, y = data['X'], data['Y'], data['cv'], data['x'], data['y']

else:
    dl = dataloader.DataLoader(args.path)
    x, y = dl.load_xyz()
    X, Y = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.target)
    if args.cluster_var == args.target: 
        cv = Y
    else:
        _, cv = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.cluster_var) 
   
    np.savez(dfpath, X=X, Y=Y, cv=cv, x=x, y=y)
    print(f"output file {dfpath}")

print(X.shape, cv.shape)
num_timesteps = cv.shape[0] // args.window * args.window

mins = 1E6
#Xout = np.zeros((num_timesteps, args.num_samples, 2))
#print(Xout.shape)

indices = np.random.choice(X.shape[1], int(args.hybrid*args.num_samples), replace=False)
Xout = X[:num_timesteps, indices, :]
print(Xout.shape)

if args.field_prediction_type == FPT_GLOBAL: # global quantity prediction
    Yout = np.zeros((num_timesteps, 1))
else: # local field prediction
    Yout = np.zeros((num_timesteps, args.num_samples))

for timestep in range(0, num_timesteps - args.window, args.window):

    print(f"\nTIMESTEP: {timestep}-{timestep + args.window}\n")

    # K-means clustering
    data = cv[timestep, :].reshape(-1, 1)
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0)
    kmeans.fit(data)
    print(args.num_clusters, kmeans.inertia_) # for creating elbow plot
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    y_pred = kmeans.predict(data)
    print(y_pred.shape)

    if args.plot:
        plt.figure(figsize=figsize)
        plt.scatter(x, y, c=kmeans.labels_, marker='.', cmap='tab10')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'KMeans clustering of {args.cluster_var}')
        plt.colorbar()
        plt.savefig(os.path.join(PLTDIR, f'kmeans_{timestep:04d}.png'), dpi=100)

    clusters = [data[np.argwhere(y_pred == i).flatten()] for i in range(args.num_clusters)]
    clusters = [cluster.flatten() for cluster in clusters]
    #print(clusters)

    # Initialize a list to store your probability distributions and their bin edges
    prob_dists = []
    bin_edges_list = []

    # Specify a consistent bin range and count
    bin_range = (np.min([np.min(cluster) for cluster in clusters]), 
                 np.max([np.max(cluster) for cluster in clusters]))
    num_bins = 50  # or choose another suitable value

    # Create probability distribution of entire plane
    counts, bin_edges = np.histogram(data, bins=num_bins, range=bin_range, density=False)
    global_prob_dist = counts / np.sum(counts)

    samples_per_cluster = []
    for cluster in clusters:
        counts, bin_edges = np.histogram(cluster, bins=num_bins, range=bin_range, density=False)
        samples_per_cluster.append(np.sum(counts))
        prob_dist = counts / np.sum(counts)
        prob_dists.append(prob_dist)
        bin_edges_list.append(bin_edges)

    n_dists = args.num_clusters

    # Compute adjacency matrix containing relative entropy for each pair of distributions
    adj_matrix = np.zeros((n_dists, n_dists))

    for i in range(n_dists):
        for j in range(n_dists):
            p = prob_dists[i] + 1e-10 # to avoid division by zero
            q = prob_dists[j] + 1e-10 # to avoid division by zero
            adj_matrix[i, j] = scipy.stats.entropy(p, q)

    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
    
    total_entropy = np.sum(adj_matrix)
    stdev_entropy = np.std(adj_matrix)
    print(f"total entropy: {total_entropy}, stdev: {stdev_entropy}")

    df = pd.DataFrame(adj_matrix)
    print(df)

    # Create a graph from the adjacency matrix and compute the minimum cut
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())

    # Select top clusters according to cutoff_threshold
    in_strengths = np.sum(adj_matrix, axis=0)
    out_strengths = np.sum(adj_matrix, axis=1)
    # for verification
    #in_strengths = dict(G.in_degree(weight='weight')).values()
    #out_strengths = dict(G.out_degree(weight='weight')).values()
    print("in-strengths:", in_strengths)
    print("out-strengths:", out_strengths)

    top_instrength = np.argsort(in_strengths)
    sorted_instrength = in_strengths[top_instrength]
    print("sorted in_strength:", sorted_instrength)
    sorted_instrength_probs = sorted_instrength / np.sum(sorted_instrength)
    print("sorted in_strength_probs:", sorted_instrength_probs)

    # linearly scale probabilities of clusters from 0.01 to 0.99
    scaled_probs = scale_probabilities(sorted_instrength_probs)
    print("scaled probs:", scaled_probs)

    top_outstrength = np.argsort(out_strengths) 

    # control vs effect - perturbation vs effect
    print(top_instrength)  # id's of clusters sorted based on in_strength
    #print(top_outstrength)

    print("number of samples per cluster: ", np.array(samples_per_cluster)[top_instrength])

    threshold = args.cutoff * np.sum(in_strengths) # same as when computing with out_strength
    print(threshold)

    sumstrength = 0
    i = len(in_strengths) - 1
    optimal_subset = []
   
    while sumstrength < threshold:
        optimal_subset.append(top_instrength[i])
        sumstrength += in_strengths[top_instrength[i]]
        i -= 1

    print('optimal subset of clusters:', optimal_subset)

    #sorted_prob_dists = [prob_dists[i] for i in top_instrength[::-1]]

    if args.plot:
        plt.clf()
        colors_ = plt.cm.get_cmap('tab10', args.num_clusters)  

        fig, ax = plt.subplots(figsize=(9, 6))
        for i, (prob_dist, bin_edges) in enumerate(zip(prob_dists, bin_edges_list)):
            alpha = 0.7 if i in optimal_subset else 0.2
            hatch = "*" if i in optimal_subset else "/"
            ax.bar(bin_edges[:-1], prob_dist, width=np.diff(bin_edges), align="edge", alpha=alpha, \
                                   label=f'Cluster {i + 1}', color=colors_(i), hatch=hatch)
        ax.bar(bin_edges[:-1], global_prob_dist, width=np.diff(bin_edges), 
               color='black', align="edge", alpha=1.0, label="Pre-clustered")
        #ax.set_title('Probability Distributions')
        ax.set_xlabel(f'Cluster variable ({args.cluster_var})')
        ax.set_ylabel('Frequency')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(PLTDIR, f'prob_dist_allin1_{timestep:04d}.png'), dpi=100)

        # Create a grid of probability distribution subplots
        #grid_size = math.ceil(math.sqrt(len(clusters)))
        #fig, axes = plt.subplots(grid_size, grid_size, figsize=(9, 9))
        #axes = axes.flatten()
        #for i, (prob_dist, bin_edges, ax) in enumerate(zip(prob_dists, bin_edges_list, axes)):
        #    ax.bar(bin_edges[:-1], prob_dist, width=np.diff(bin_edges), align="edge", alpha=0.5)
        #    ax.bar(bin_edges[:-1], global_prob_dist, width=np.diff(bin_edges), align="edge", alpha=0.5)
        #    ax.set_title(f'Cluster {i + 1}')
        #for ax in axes[len(clusters):]: ax.remove()
        #plt.tight_layout()
        #plt.savefig(os.path.join(PLTDIR, f'prob_dist_{timestep:04d}.png'), dpi=100)

    num_samples = len(kmeans.labels_)
    mask = np.isin(kmeans.labels_, optimal_subset)
    id_subsample = np.arange(X.shape[1])[mask]
    label_subsample = kmeans.labels_[mask]
    num_samples_compressed = len(label_subsample)
    print(f"uncompressed samples: {num_samples}, filtered subset: {num_samples_compressed},", 
          f"compression factor: {num_samples / num_samples_compressed:.1f}X")
    mins = min(mins, num_samples_compressed)

    # Randomly sample from the optimal clusters
    #indices = np.random.choice(subsampled_Y.shape[0], args.num_samples, replace=False)
    #print("***", subsampled_X.shape, args.num_samples, id_subsample.shape)

    if args.subsample == "random":
        indices1 = np.random.choice(num_samples_compressed, args.num_samples, replace=False)
        indices2 = id_subsample[indices1]

    elif args.subsample == "random-weighted":
        probs = np.abs(np.squeeze(data[mask]))
        probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs)) 
        probs /= np.sum(probs)
        indices1 = np.random.choice(num_samples_compressed, args.num_samples, replace=False, p=probs)
        indices2 = id_subsample[indices1]

    elif args.subsample == "silhouette":
        sample_silhouette_values = silhouette_samples(data, cluster_labels)
        probs = np.zeros((num_samples_compressed))
        #print("***", probs.shape, id_subsample.shape)
        for i in optimal_subset:
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            cluster_silhouette_values = (cluster_silhouette_values - np.min(cluster_silhouette_values)) / \
                                        (np.max(cluster_silhouette_values) - np.min(cluster_silhouette_values))
            #cluster_silhouette_values = np.abs(cluster_silhouette_values)
            #print(len(cluster_silhouette_values), len(probs[label_subsample == i]), i, label_subsample)
            probs[label_subsample == i] = cluster_silhouette_values
        probs /= np.sum(probs)
        non_zeros = np.count_nonzero(probs)
        if np.count_nonzero(probs) < args.num_samples:
            raise ValueError(f"decrease --num_samples to be less or equal to {non_zeros}")
        indices1 = np.random.choice(num_samples_compressed, args.num_samples, replace=False, p=probs)
        indices2 = id_subsample[indices1]

    else:
        raise ValueError(f"must specify subsampling method using --subsample")
    
    ts = timestep 
    for sub_timestep in range(args.window):
        if args.verbose: print(f"timestep: {ts}")
        
        # Find the indices of the original dataset, data, that have optimal clusters
        subsampled_X = X[ts, mask, :]
        subsampled_Y = Y[ts] if args.field_prediction_type == FPT_GLOBAL else Y[ts, mask]

        if args.field_prediction_type == FPT_GLOBAL:
            subsampled_X = subsampled_X[indices1, :]
        else:
            subsampled_X, subsampled_Y = subsampled_X[indices1, :], subsampled_Y[indices1]

        if args.verbose: print(subsampled_X.shape, subsampled_Y.shape)

        Xout[ts, :args.num_samples, :] = subsampled_X
        Yout[ts, :args.num_samples] = subsampled_Y

        if args.plot:
            plt.clf()
            plt.figure(figsize=figsize)
            plt.scatter(x[indices2], y[indices2], c=kmeans.labels_[indices2], marker='.', \
                        cmap='tab10', vmin=-0.5, vmax=max(kmeans.labels_) + 0.5)
            plt.xlim([-20, 60])
            plt.ylim([-10, 10])
            plt.savefig(os.path.join(PLTDIR, f'frame_{ts:04d}_{args.subsample}.png'), dpi=100)

        ts += 1

    #df = pd.DataFrame(np.concatenate((subsampled_Y, subsampled_X), axis=1), columns=[args.target, 'u', 'v'])
    #df.to_csv(f"data_{timestep:05}.csv", index=False)

    # Show only optimal clusters
    #if args.plot:
    #    # set clusters below threshold to -1 
    #    kmeans.labels_[~mask] = -1 
    #    # and set their color to white so they won't be visible
    #    cmap_white_first = colors.ListedColormap(['white', *plt.cm.viridis.colors])
    #    plt.figure(figsize=figsize)
    #    plt.scatter(x, y, c=kmeans.labels_, marker='.', cmap=cmap_white_first, \
    #                vmin=-0.5, vmax=max(kmeans.labels_) + 0.5)
    #    plt.xlabel('X')
    #    plt.ylabel('Y')
    #    plt.title('Features with highest entropy')
    #    cbar = plt.colorbar(ticks=np.arange(0, max(kmeans.labels_), 1))
    #    cbar.set_label('Cluster Label')
    #    plt.savefig(os.path.join(PLTDIR, f'frame_{timestep:04d}.png'), dpi=100)

print(Xout.shape, Yout.shape)
np.savez(os.path.join(SNPDIR, 'subsampled.npz'), X=Xout, Y=Yout, target=args.target)
print('min number of samples over all timesteps:', mins)
