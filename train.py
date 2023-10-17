import importlib
import numpy as np
import os
import tensorflow as tf

from gauss_rank_scaler import GaussRankScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import dataloader
from args import args
from constants import *
from helpers import tune, scale, print_stats, plot_histograms

data = np.load(os.path.join(SNPDIR, 'subsampled.npz'))
X, Y, target = data['X'], data['Y'], data['target']

print(X.shape, Y.shape, len(Y.shape))

if args.arch == 'lstm':
    print('creating time sequences...')
    X, Y = dataloader.create_sequences(X, Y, window_size=args.window, \
                                       field_prediction_type=args.field_prediction_type)
    print(X.shape, Y.shape)
    num_sequences, sequence_length, num_features = X.shape
    num_samples = X.shape[0]
    if args.field_prediction_type == FPT_GLOBAL:
        Y = Y.reshape(num_sequences, sequence_length)
else:
    Y = np.squeeze(Y)

print('Data shape for network:')
print(X.shape, Y.shape)

# Split data into train/test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_frac, shuffle=False)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

if args.arch == 'fcn':
    # Flatten input so it has only two dimensions: (n_samples, n_features)
    X = X.reshape(X.shape[0], -1)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print_stats('Train', X_train, Y_train)
    print_stats('Test', X_test, Y_test)

print('train/test shapes:', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

if args.plot: plot_histograms(X_train, X_test, Y_train, Y_test)

# Scale the data 
# Did the scaler work well before? Data is supposed to be fed (n_samples, n_features)

scaler_x = eval(args.xscaler)()
X_train = scale(scaler_x.fit_transform, X_train)
X_test = scale(scaler_x.transform, X_test)

if args.yscaler not 'None':
    scaler_y = eval(args.yscaler)()
    if args.arch == 'lstm':
        Y_train = scale(scaler_y.fit_transform, Y_train)
        Y_test = scale(scaler_y.transform, Y_test)
    else:
        Y_train = scaler_y.fit_transform(Y_train.reshape(-1, 1))
        Y_test = scaler_y.transform(Y_test.reshape(-1, 1))

Y_train, Y_test = Y_train/args.yscalefactor, Y_test/args.yscalefactor

print_stats('Train', X_train, Y_train)
print_stats('Test', X_test, Y_test)

if args.plot: plot_histograms(X_train, X_test, Y_train, Y_test)

# Define model
input_shape = X[0].shape
model = importlib.import_module('archs.' + args.arch).build_model(input_shape, window=args.window)
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.7, patience=5, min_lr=1e-8)

# FCN networks can be highly noisy at initial training if lr is high.
# High lr is often desired to explore most of the solution landscape, 
# then reducing it when falling into an optimal minima troughs
early_stop = EarlyStopping(monitor='loss', patience=args.patience)
callbacks = [reduce_lr, early_stop]

# Train model
if args.tune:
    func = importlib.import_module('archs.' + args.arch).get_meta_model(input_shape)
    model = tune(func, X_train, Y_train, X_test, Y_test, \
                 batch_size=args.batch, epochs=50, max_epochs=args.epochs)
    model.build(input_shape=input_shape)
else:
    model.fit(X_train, Y_train, batch_size=args.batch, epochs=args.epochs, \
              callbacks=callbacks, validation_data=([X_test], [Y_test]))

# Evaluate the model
loss = model.evaluate(X_test, Y_test)
print(f'Loss: {loss:.04f}')

# Make prediction
#prediction = model.predict(X_test)
#print('Prediction:', prediction)

# Save model
model.save(f"models/{args.arch}/1")
np.savez(os.path.join(SNPDIR, "test.npz"), X_test=X_test, Y_test=Y_test, X_train=X_train, Y_train=Y_train)
