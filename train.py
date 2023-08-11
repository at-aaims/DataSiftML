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
    if args.local:
        X, Y = X[:, indices, :], Y[:, indices]
    else:
        X = X[:, indices, :]
        Y = np.squeeze(Y)

if args.subsample == 'none':
    Y = np.squeeze(Y)

print(X.shape, Y.shape)

# create time sequences
print('Data aggregated into sequences?: ', args.sequence)
if args.sequence:
    X, Y = dataloader.create_sequences(X, Y, window_size=args.window)

print(X.shape, Y.shape)

# reshape input_data to make it 3D: (Batch_size, timesteps, input_dim)
#num_samples, num_sequences, sequence_length, num_features = X.shape
#num_samples, num_sequences, sequence_length, num_features = X.shape
#X = X.reshape(num_samples * num_sequences, sequence_length, num_features)
#Y = Y.reshape(num_samples * num_sequences, sequence_length)
#Y = Y.reshape(num_sequences, sequence_length)
#Y = np.expand_dims(Y, axis=-1)


# split data into train/test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_frac, shuffle=False)


print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)


# scale the data
scaler_x = eval(args.scaler)()
X_train = scale(scaler_x.fit_transform, X_train)
X_test = scale(scaler_x.transform, X_test)

# define model
input_shape = X[0].shape
model = importlib.import_module('archs.' + args.arch).build_model(input_shape, window=args.window)
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

