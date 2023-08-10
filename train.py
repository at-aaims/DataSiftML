import importlib
import numpy as np
import os
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

import dataloader
from args import args
from helpers import tune, scale
from subsample import subsample_random


if args.subsample == 'maxent':
    data = np.load('subsampled.npz')
    X, Y = data['X'], np.squeeze(data['Y'])
else: 
    dl = dataloader.DataLoader(args.path)
    X, Y = dl.load_multiple_timesteps(args.write_interval, args.num_time_steps, target=args.target)

print(X.shape, Y.shape)

# subsample data
if args.subsample == 'random':
    indices = subsample_random(X, args.num_samples)
    X, Y = X[:, indices, :], Y[:, indices]
    print(X.shape, Y.shape)

# create time sequences
X, Y = dataloader.create_sequences(X, Y, window_size=args.window) 

# reshape input_data to make it 3D: (Batch_size, timesteps, input_dim)
num_samples, num_sequences, sequence_length, num_features = X.shape
X = X.reshape(num_samples * num_sequences, sequence_length, num_features)
Y = Y.reshape(num_samples * num_sequences, sequence_length)
Y = np.expand_dims(Y, axis=-1)

print(X.shape, Y.shape)

# split data into train/test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_frac, shuffle=False)

# scale the data
scaler_x = eval(args.scaler)()
X_train = scale(scaler_x.fit_transform, X_train)
X_test = scale(scaler_x.transform, X_test)

# define model
input_shape = X[0].shape
model = importlib.import_module('archs.' + args.arch).build_model(input_shape)
model.summary()

# train model
if args.tune:
    func = importlib.import_module('archs.' + args.arch).get_meta_model(input_shape)
    model = tune(func, X_train, Y_train, X_test, Y_test, \
                 batch_size=args.batch, epochs=5, max_epochs=args.epochs)
    model.build(input_shape=input_shape)
else:
    model.fit(X_train, Y_train, batch_size=args.batch, epochs=args.epochs)

# evaluate the model
loss = model.evaluate(X_test, Y_test)
print('Loss:', loss)

# make a prediction
prediction = model.predict(X_test)
#print('Prediction:', prediction)

# save model
model.save(f"models/{args.arch}/1")

