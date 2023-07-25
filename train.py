import importlib
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import dataloader
from args import args

# load data
dl = dataloader.DataLoader(args.path)
X, Y = dataloader.create_sequences(*dl.load_multiple_timesteps(100, 100))
print(X.shape, Y.shape)

## reshape input_data to make it 3D: (Batch_size, timesteps, input_dim)
num_samples, num_sequences, sequence_length, num_features = X.shape
X = X.reshape(num_samples * num_sequences, sequence_length, num_features)
Y = Y.reshape(num_samples * num_sequences, sequence_length)
Y = np.expand_dims(Y, axis=-1)

print(X.shape, Y.shape)

# define model
model = importlib.import_module('archs.' + args.arch).build_model(X[0].shape)
model.summary()

# compile model
model.compile(loss='mae', optimizer='adam')

# train model
model.fit(X, Y, batch_size=args.batch, epochs=args.epochs)

# save model
model.save(f"{args.arch}/1")
