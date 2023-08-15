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
from constants import *
from helpers import tune, scale


data = np.load(os.path.join(SNPDIR, 'subsampled.npz'))
X, Y, target = data['X'], np.squeeze(data['Y']), data['target']

print(X.shape, Y.shape)

if args.subsample == 'none':
    Y = np.squeeze(Y)

print(X.shape, Y.shape)

# create time sequences
print('Data aggregated into sequences?: ', args.sequence)

if args.sequence:
    X, Y = dataloader.create_sequences(X, Y, window_size=args.window, field_prediction_type=args.field_prediction_type)
    print(X.shape)
    print(Y.shape)
    num_sequences, sequence_length, num_features = X.shape
    num_samples = X.shape[0]
    if args.field_prediction_type == FPT_GLOBAL:
        Y = Y.reshape(num_sequences, sequence_length)

print('Data shape for network:')
print(X.shape, Y.shape)

# split data into train/test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_frac, shuffle=False)

if args.arch == 'fcn':
    # Flattening input so it has only two dimensions: (n_samples, n_features)
    # n_features = n_points*n_variables
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
    print('########## Data Statistics ###########')
    print('Train U')
    print('Max value: {}'.format(np.amax(X_train[:,0])))
    print('Min value: {}'.format(np.amin(X_train[:,0])))
    print('Mean value: {}'.format(np.mean(X_train[:,0])))
    print('#########################')
    print('Train V')
    print('Max value: {}'.format(np.amax(X_train[:,1])))
    print('Min value: {}'.format(np.amin(X_train[:,1])))
    print('Mean value: {}'.format(np.mean(X_train[:,1])))
    print('#########################')
    print('Train Drag')
    print('Max value: {}'.format(np.amax(Y_train[:])))
    print('Min value: {}'.format(np.amin(Y_train[:])))
    print('Mean value: {}'.format(np.mean(Y_train[:])))
    print('#########################')
    print('Test U')
    print('Max value: {}'.format(np.amax(X_test[:,0])))
    print('Min value: {}'.format(np.amin(X_test[:,0])))
    print('Mean value: {}'.format(np.mean(X_test[:,0])))
    print('#########################')
    print('Test V')
    print('Max value: {}'.format(np.amax(X_test[:,1])))
    print('Min value: {}'.format(np.amin(X_test[:,1])))
    print('Mean value: {}'.format(np.mean(X_test[:,1])))
    print('#########################')
    print('Test Drag')
    print('Max value: {}'.format(np.amax(Y_test[:])))
    print('Min value: {}'.format(np.amin(Y_test[:])))
    print('Mean value: {}'.format(np.mean(Y_test[:])))
    print('#########################')

print('Train Shape')
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('Test Shape:')
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

# scale the data
# Did the scaler worked well before? Data is supposed to be fed (n_samples, n_features)
scaler_x = eval(args.scaler)()
X_train = scale(scaler_x.fit_transform, X_train)
Y_train = Y_train/10
X_test = scale(scaler_x.transform, X_test)
Y_test = Y_test/10

print('########## Data Statistics AFTER SCALING ###########')
print('Train U')
print('Max value: {}'.format(np.amax(X_train[:,0])))
print('Min value: {}'.format(np.amin(X_train[:,0])))
print('Mean value: {}'.format(np.mean(X_train[:,0])))
print('#########################')
print('Train V')
print('Max value: {}'.format(np.amax(X_train[:,1])))
print('Min value: {}'.format(np.amin(X_train[:,1])))
print('Mean value: {}'.format(np.mean(X_train[:,1])))
print('#########################')
print('Train Drag')
print('Max value: {}'.format(np.amax(Y_train[:])))
print('Min value: {}'.format(np.amin(Y_train[:])))
print('Mean value: {}'.format(np.mean(Y_train[:])))
print('#########################')
print('Test U')
print('Max value: {}'.format(np.amax(X_test[:,0])))
print('Min value: {}'.format(np.amin(X_test[:,0])))
print('Mean value: {}'.format(np.mean(X_test[:,0])))
print('#########################')
print('Test V')
print('Max value: {}'.format(np.amax(X_test[:,1])))
print('Min value: {}'.format(np.amin(X_test[:,1])))
print('Mean value: {}'.format(np.mean(X_test[:,1])))
print('#########################')
print('Test Drag')
print('Max value: {}'.format(np.amax(Y_test[:])))
print('Min value: {}'.format(np.amin(Y_test[:])))
print('Min value: {}'.format(np.mean(Y_test[:])))
print('#########################')


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
#prediction = model.predict(X_test)
#print('Prediction:', prediction)

# save model
model.save(f"models/{args.arch}/1")

