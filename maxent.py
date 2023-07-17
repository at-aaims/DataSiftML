import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluidfoam import readscalar, readvector, readforce
from sklearn.cluster import KMeans, Birch

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=10, help="number of clusters")
parser.add_argument("--path", type=str, default='.', help="path to simulation")
parser.add_argument("--time", type=str, default='1000', help="time step to analyze")
parser.add_argument("--plot", action='store_true', default=False, help="show plots")
parser.add_argument("--verbose", action='store_true', default=False, help="verbose output")
args = parser.parse_args()

# Read solution values from OpenFOAM simulation
p = readscalar(args.path, args.time, 'p.gz')
x, y, z = readvector(args.path, args.time, 'C.gz')
Ux, Uy, Uz = readvector(args.path, args.time, 'U.gz')
forces = readforce(args.path, time_name='0', name='forces')
drag = forces[:, 1] + forces[:, 2]

if args.verbose:
    print('force.shape:', forces.shape)
    print('drag.shape:', drag.shape)
    print(drag)

if args.plot:
    plt.figure()
    plt.plot(forces[:, 0], drag)
    plt.xlabel('t')
    plt.ylabel('drag')
    plt.ylim(-10, 10) 
    plt.title('Drag history')
    plt.show()

# Add an extra dimension
p = np.expand_dims(p, axis=1)
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)
Ux = np.expand_dims(Ux, axis=1)
Uy = np.expand_dims(Uy, axis=1)

stacked = np.hstack((x, y, p, Ux, Uy))

if args.verbose:
    print(p.shape)
    print(x.shape)
    print(y.shape)
    print(Ux.shape)
    print(Uy.shape)
    print(stacked.shape)

df = pd.DataFrame(stacked, columns=['x', 'y', 'p', 'Ux', 'Uy'])

# Output CSV file named by timestamp, e.g. 1000.csv
file_name = args.time + '.csv'
df.to_csv(file_name, index=False)

with open(file_name, 'a') as file:
    drag_value = drag[int(args.time)]
    file.write(f"# Drag: {drag_value}\n")

# Create a KMeans instance
kmeans = KMeans(n_clusters=args.n, random_state=0)

# Fit the model to your data
kmeans.fit(stacked[:, 2:5])

# Predict the cluster labels of the data
labels = kmeans.predict(stacked[:, 2:5])

# Print the cluster centers
if args.verbose:
    print("Cluster centers:")
    print(kmeans.cluster_centers_)

if args.plot:
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

