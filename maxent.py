import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluidfoam import readscalar, readvector
from sklearn.cluster import KMeans, Birch

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=10, help="number of clusters")
args = parser.parse_args()

path = '.'
time = '1000'
p = readscalar(path, time, 'p.gz')
x, y, z = readvector(path, time, 'C.gz')
Ux, Uy, Uz = readvector(path, time, 'U.gz')

# Add an extra dimension
p = np.expand_dims(p, axis=1)
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)
Ux = np.expand_dims(Ux, axis=1)
Uy = np.expand_dims(Uy, axis=1)

print(p.shape)
print(x.shape)
print(y.shape)
print(Ux.shape)
print(Uy.shape)

stacked = np.hstack((x, y, p, Ux, Uy))
print(stacked.shape)

df = pd.DataFrame(stacked, columns=['x', 'y', 'p', 'Ux', 'Uy'])

print(df)
df.to_csv(time + '.csv', index=False)

# Create a KMeans instance
kmeans = KMeans(n_clusters=args.n, random_state=0)

print(stacked.shape)

# Fit the model to your data
kmeans.fit(stacked[:, 2:5])

# Predict the cluster labels of the data
labels = kmeans.predict(stacked[:, 2:5])

# Print the cluster centers
print("Cluster centers:")
print(kmeans.cluster_centers_)

#plt.contourf(x, y, kmeans.labels_, cmap='viridis')
plt.scatter(x, y, c=kmeans.labels_, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KMeans Clustering')
plt.show()


# perform stratified sampling



# BIRCH
#birch = Birch(threshold=0.1, branching_factor=50)
#birch.fit(stacked)
#labels = birch.predict(stacked)
#print("Cluster centers:")
#print(birch.subcluster_centers_)

