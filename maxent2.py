"""This script computes MaxEnt using descriptor-based technique"""

import dataloader
import numpy as np
import pandas as pd

from args import args
from sklearn.cluster import KMeans

dl = dataloader.DataLoader(args.path)

write_interval = 100
num_time_steps = 100

X, Y = dl.load_multiple_timesteps(write_interval, num_time_steps)

# extract data at a single point for all times
pid = 27
X, Y = X[:, pid, :], Y[:, pid]
print(X.shape, Y.shape)

window = 4

# window the data by window size
num_clusters = X.shape[0] // window
print(f"num_clusters: {num_clusters}")
X = X.reshape(-1, window, X.shape[1])
Y = Y.reshape(-1, window)

print(X.shape, Y.shape)

times = np.linspace(0, window-1, window)

var = 0

slopes = np.zeros(num_clusters)
for i in range(num_clusters):
    slopes[i], _ = np.polyfit(X[i,:,var], times, deg=1)

data = {
    'means': np.mean(X[:,:,var], axis=1),
    'stdevs': np.std(X[:,:,var], axis=1),
    'curvdists': np.linalg.norm(X[:,:,var], axis=1),
    'slopes': slopes
}

df = pd.DataFrame(data)
print(df)

# K-means clustering
kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
kmeans.fit(df)
df['cluster'] = kmeans.predict(df)
print(df['cluster'])