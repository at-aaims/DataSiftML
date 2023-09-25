import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from args import args
from constants import SNPDIR, PLTDIR
from matplotlib.colors import LinearSegmentedColormap

data = np.load('snapshots/subsampled.npz')
x, y = data['x'], data['y']

data = np.load(os.path.join(SNPDIR, 'test_maxent.npz'))
X_test, Y_test, X_train, Y_train = data['X_test'], data['Y_test'], data['X_train'], data['Y_train']

# Load the saved model
model = tf.keras.models.load_model(f"models/{args.arch}/1")

# Ensure the model has been loaded correctly
model.summary()

print(X_test.shape, Y_test.shape, X_train.shape, Y_train.shape)

for i in range(X_test.shape[0]):
    ypred = model.predict(X_test)
    errors = np.abs(ypred - Y_test) / ypred.shape[0]
    print(errors.shape)

# Define a green-yellow-red colormap
cmap_colors = [(0, "green"), (0.5, "yellow"), (1, "red")]
cmap_gyr = LinearSegmentedColormap.from_list("GreenYellowRed", cmap_colors)

#emin = np.min(errors)
#emax = np.max(errors)
emin = 0
emax = 0.015

for t in range(X_test.shape[0]):
    plt.clf() 
    plt.figure(figsize=(9, 2))
    plt.scatter(x, y, c=errors[t, :], marker='.', cmap=cmap_gyr, vmin=emin, vmax=emax)
    plt.xlim([-25, 65])
    plt.ylim([-10, 10])
    plt.axis('equal') 
    cbar = plt.colorbar(ticks=np.linspace(emin, emax, 5))
    cbar.set_label(r'L1 error')
    plt.savefig(os.path.join(PLTDIR, f'errors_{t:04d}_random.png'), dpi=100)
