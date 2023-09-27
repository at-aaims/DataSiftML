"""This script computes MaxEnt using Murali's approach with a collective variable
   such as vorticity, pressure, or using the input features such as flow field [u,v]."""

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


figsize = (9, 2)

dfpath = os.path.join(SNPDIR, DRAWFN)
dfpath2 = os.path.join(SNPDIR, 'interpolated.npz')

dl = dataloader.DataLoader(args.path)
Xold, Y = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.target)
np.savez(dfpath, Y=Y)

if os.path.exists(dfpath2):
    print('Reading custom data')
    unsdata = np.load(dfpath)
    Y = unsdata['Y']
    print('drag',Y.shape)
    data = np.load(dfpath2)
    print(list(data.keys()))
    #x, y, X, Yp, Wz = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'] #, data['y']
    #X, Y, cv = data['X'], data['Y'], data['cv'] # , data['x'], data['y']
    x, y, X, Yp, cv = data['x'], data['y'], data['X'], data['Y'], data['cv']

#    if args.cluster_var == args.target: 
#        cv = Y
#    else:
#        _, cv = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.cluster_var) 
#   
# Load all data
#X = np.load('./data/VTKnpy/U_100x100.npy')
#x = np.load('./data/VTKnpy/x_100x100.npy')
#print(x.shape)
#y = np.load('./data/VTKnpy/y_100x100.npy')
#print(y.shape)
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2],X.shape[-1])
X = X[1:]
print(X.shape)

Yp = Yp.reshape(Yp.shape[0],Yp.shape[1]*Yp.shape[2],Yp.shape[-1])
#cv = Y
#cv = np.squeeze(cv)
print('Pressure flattened for clustering: {}'.format(Yp.shape))

x = x[1:]
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2],1)
print(x.shape)

y = y[1:]
y = y.reshape(y.shape[0],y.shape[1]*y.shape[2],1)
print(y.shape)

#print('X shape {} | cv shape {}'.format(X.shape,cv.shape))
num_timesteps = cv.shape[0] // args.window * args.window

mins = 1E6

if args.subsample == "equal": 
    num_samples_per_cluster = args.num_samples
    args.num_samples *= args.num_clusters

Xout = np.zeros((num_timesteps, args.num_samples, 2))

if args.field_prediction_type == FPT_GLOBAL: # global quantity prediction
    print('Predicting Drag')
    Yout = np.zeros((num_timesteps, 1))
else: # local field prediction
    Yout = np.zeros((num_timesteps, args.num_samples))

for timestep in range(0, num_timesteps - args.window, args.window):

    print(f"\nTIMESTEP: {timestep}-{timestep + args.window}\n")

    # K-means clustering
    indata = X[timestep, :,0].reshape(-1, 1)
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0)
    kmeans.fit(indata)
    print(args.num_clusters, kmeans.inertia_) # for creating elbow plot
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    y_pred = kmeans.predict(indata)
    print('y_pred shape',y_pred.shape)
    print(y_pred)
    print(np.amax(y_pred))

    if args.plot:
        plt.figure(figsize=figsize)
        plt.scatter(x[0], y[0], c=kmeans.labels_, marker='.', cmap='tab10')
        plt.xlabel('Domain x-coordinate')
        plt.ylabel('Domain y-coordinate')
        plt.title(f'KMeans clustering of {args.cluster_var}')
        plt.colorbar()
        plt.savefig(os.path.join(PLTDIR, f'kmeans_{timestep:04d}.png'), dpi=100)

    clusters = [indata[np.argwhere(y_pred == i).flatten()] for i in range(args.num_clusters)]
    #print('Got here 2')
    #print(len(clusters[0]))
    #print(len(clusters))
    #print(clusters[0])
    #print(clusters[8])
    clusters = [cluster for cluster in clusters]

    # Initialize a list to store your probability distributions and their bin edges
    prob_dists = []
    bin_edges_list = []

    # Specify a consistent bin range and count
    bin_range = (np.min([np.min(cluster) for cluster in clusters]), 
                 np.max([np.max(cluster) for cluster in clusters]))
    num_bins = 50  # or choose another suitable value

    # Create probability distribution of entire plane
    counts, bin_edges = np.histogram(indata, bins=num_bins, range=bin_range, density=False)
    global_prob_dist = counts / np.sum(counts)

    samples_per_cluster = []
    for cluster in clusters:
        counts, bin_edges = np.histogram(cluster, bins=num_bins, range=bin_range, density=False)
        samples_per_cluster.append(np.sum(counts))
        prob_dist = counts / np.sum(counts)
        prob_dists.append(prob_dist)
        bin_edges_list.append(bin_edges)

    print("*** counts: ", samples_per_cluster)

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

    if args.plot:
        plt.clf()
        #plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(12,10),facecolor='1') 
        plt.imshow(adj_matrix, cmap='inferno')
        cbar = plt.colorbar(); cbar.set_label(r'relative entropy, $D$')
        plt.axis('equal')
        plt.savefig(os.path.join(PLTDIR, f'adj_matrix_{timestep:04d}.png'), dpi=100)

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

    #sorted_prob_dists = [prob_dists[i] for i in top_instrength[::-1]]

    if args.plot:
        plt.clf()
        colors_ = plt.cm.get_cmap('tab10', args.num_clusters)  

        fig, ax1 = plt.subplots(figsize=(9, 6))
        for i, (prob_dist, bin_edges) in enumerate(zip(prob_dists, bin_edges_list)):
            alpha = 0.7
            ax1.bar(bin_edges[:-1], prob_dist, width=np.diff(bin_edges), align="edge", alpha=alpha,
                    label=f'Cluster {i + 1} ({samples_per_cluster[i]})', color=colors_(i))

        ax2 = ax1.twinx()
        ax2.bar(bin_edges[:-1], global_prob_dist, width=np.diff(bin_edges),
                color='black', align='edge', alpha=0.2, label='Pre-clustered', edgecolor='red', linewidth=1)
        ax2.set_ylabel('Pre-clustered frequency')
        ax1.set_xlabel(f'Cluster variable ({args.cluster_var})')
        ax1.set_ylabel('Clustered Frequency')

        # Retrieve the legend handles and labels from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        # Combine the legend entries and show them in a single legend
        ax1.legend(handles1 + handles2, labels1 + labels2)

        plt.tight_layout()
        plt.savefig(os.path.join(PLTDIR, f'prob_dists_{timestep:04d}.png'), dpi=100)

    num_samples = len(kmeans.labels_)

    if args.subsample == "proportional":
        # probabilistically select from clusters according to in-strength values
        probs = np.zeros((indata.shape[0]))
        for i in range(args.num_clusters):
            probs[cluster_labels == i] = in_strengths[i]

        probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
        probs /= np.sum(probs)

        indices1 = np.random.choice(indata.shape[0], args.num_samples, replace=False, p=probs)
        indices2 = np.copy(indices1)

        if args.plot:
            plt.clf()
            plt.hist(cluster_labels[indices1], bins=args.num_clusters, edgecolor='k')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram')
            plt.savefig(os.path.join(PLTDIR, f'histogram_{timestep:04d}.png'))

    elif args.subsample == "equal": 

        indices1 = np.zeros((num_samples_per_cluster * args.num_clusters), dtype=int)
        for i in range(args.num_clusters):
            mins = min(mins, samples_per_cluster[i])
            temp_indices = np.where(cluster_labels == i)[0]
            indices1[i*num_samples_per_cluster:(i+1)*num_samples_per_cluster] = \
                     np.random.choice(temp_indices, num_samples_per_cluster, replace=False)
        indices2 = np.copy(indices1)

    elif args.subsample == "equalpercentage": 

        percentage = args.cutoff
        indices1 = np.zeros((1), dtype=int)
        for i in range(args.num_clusters):
            mins = min(mins, samples_per_cluster[i])
            temp_indices = np.where(cluster_labels == i)[0]
            indices1 = np.append(indices1, np.random.choice(temp_indices, \
                                 int(percentage * len(temp_indices)), replace=False))
        indices1 = indices1[1:]
        indices1 = np.random.choice(indices1, args.num_samples, replace=False)
        indices2 = np.copy(indices1)

    else: # rest all only select top clusters for analysis

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
        print(top_outstrength)

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

        mask = np.isin(kmeans.labels_, optimal_subset)
        id_subsample = np.arange(X.shape[1])[mask]
        label_subsample = kmeans.labels_[mask]
        num_samples_compressed = len(label_subsample)
        print(f"uncompressed samples: {num_samples}, filtered subset: {num_samples_compressed},", 
              f"compression factor: {num_samples / num_samples_compressed:.1f}X")
        mins = min(mins, num_samples_compressed)

        if args.subsample == "random":
            indices1 = np.random.choice(num_samples_compressed, args.num_samples, replace=False)
            indices2 = id_subsample[indices1]

        elif args.subsample == "random-weighted": # use cluster var as weighting parameter
            probs = np.abs(np.squeeze(indata[mask])) 
            probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs)) 
            probs /= np.sum(probs)
            indices1 = np.random.choice(num_samples_compressed, args.num_samples, replace=False, p=probs)
            indices2 = id_subsample[indices1]

        elif args.subsample == "silhouette": # use silhouette values as weighting parameter
            sample_silhouette_values = silhouette_samples(indata, cluster_labels)
            probs = np.zeros((num_samples_compressed))
            for i in optimal_subset:
                cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                cluster_silhouette_values = (cluster_silhouette_values - np.min(cluster_silhouette_values)) / \
                                            (np.max(cluster_silhouette_values) - np.min(cluster_silhouette_values))
                probs[label_subsample == i] = cluster_silhouette_values
            probs /= np.sum(probs)
            non_zeros = np.count_nonzero(probs)
            if np.count_nonzero(probs) < args.num_samples:
                raise ValueError(f"decrease --num_samples to be less or equal to {non_zeros}")
            indices1 = np.random.choice(num_samples_compressed, args.num_samples, replace=False, p=probs)
            indices2 = id_subsample[indices1]
    
    ts = timestep 
    for sub_timestep in range(args.window):
        if args.verbose: print(f"timestep: {ts}")
        
        if args.subsample in ["proportional", "equal", "equalpercentage"]:
            subsampled_X = X[ts, indices1, :]
            subsampled_Y = Y[ts] if args.field_prediction_type == FPT_GLOBAL else Y[ts, indices1]
        else:
            # Find the indices of the original dataset, data, that have optimal clusters
            subsampled_X = X[ts, mask, :]
            subsampled_Y = Y[ts] if args.field_prediction_type == FPT_GLOBAL else Y[ts, mask]

            if args.field_prediction_type == FPT_GLOBAL:
                subsampled_X = subsampled_X[indices1, :]
            else:
                subsampled_X, subsampled_Y = subsampled_X[indices1, :], subsampled_Y[indices1]

        if args.verbose: print(subsampled_X.shape, subsampled_Y.shape)

        Xout[ts, :, :] = subsampled_X
        Yout[ts, :] = subsampled_Y

        ts += 1

    if args.plot:
        plt.clf()
        plt.figure(figsize=figsize)
        print(x[0].shape)
        print('shape of x grid')
        plt.scatter(x[0][indices2], y[0][indices2], c=kmeans.labels_[indices2], marker='.', \
                    cmap='tab10', vmin=-0.5, vmax=max(kmeans.labels_) + 0.5)
        plt.xlim([-25, 65])
        plt.ylim([-10, 10])
        plt.axis('equal')
        plt.savefig(os.path.join(PLTDIR, f'frame_{ts:04d}_{args.subsample}.png'), dpi=100)

print(Xout.shape, Yout.shape)
np.savez(os.path.join(SNPDIR, 'subsampled.npz'), X=Xout, Y=Yout, target=args.target)
if args.subsample != "proportional": print('min number of samples over all timesteps:', mins)
