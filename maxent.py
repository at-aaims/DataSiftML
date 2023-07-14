import numpy as np
import pandas as pd

from fluidfoam import readscalar, readvector
from sklearn.cluster import KMeans, Birch

sol = '.'
timename = '1000'
p = readscalar(sol, timename, 'p.gz')
Ux, Uy, Uz = readvector(sol, timename, 'U.gz')

# Add an extra dimension
p = np.expand_dims(p, axis=1)
Ux = np.expand_dims(Ux, axis=1)
Uy = np.expand_dims(Uy, axis=1)

print(p.shape)
print(Ux.shape)
print(Uy.shape)

stacked = np.hstack((p, Ux, Uy))
print(stacked.shape)

df = pd.DataFrame(stacked)
print(df)
df.to_csv("output.csv")

# Create a KMeans instance
kmeans = KMeans(n_clusters=3, random_state=0)

# Fit the model to your data
kmeans.fit(stacked)

# Predict the cluster labels of the data
labels = kmeans.predict(stacked)

# Print the cluster centers
print("Cluster centers:")
print(kmeans.cluster_centers_)



# Create a BIRCH instance
birch = Birch(threshold=0.1, branching_factor=50)

# Fit the model to your data
birch.fit(stacked)

# Predict the cluster labels of the data
labels = birch.predict(stacked)

# Print the cluster centers
print("Cluster centers:")
print(birch.subcluster_centers_)

