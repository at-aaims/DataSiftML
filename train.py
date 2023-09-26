import importlib
import numpy as np
import os
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import dataloader
from args import args
from constants import *
from helpers import tune, scale, print_stats

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

# split data into train/test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_frac, shuffle=False)

if args.arch == 'fcn':
    # Flattening input so it has only two dimensions: (n_samples, n_features)
    # n_features = n_points*n_variables
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
    print_stats(X_train, Y_train, X_test, Y_test)

print('train/test shapes:', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# scale the data
# Did the scaler worked well before? Data is supposed to be fed (n_samples, n_features)
scaler_x = eval(args.scaler)()
X_train = scale(scaler_x.fit_transform, X_train)
X_test = scale(scaler_x.transform, X_test)
Y_train, Y_test = Y_train/10, Y_test/10
print_stats(X_train, Y_train, X_test, Y_test)

# define model
input_shape = X[0].shape
model = importlib.import_module('archs.' + args.arch).build_model(input_shape, window=args.window)
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.7, patience=2, min_lr=1e-8)

# FCN networks can be highly noisy at initial training if lr is high.
# High lr is often desired to explore most of the solution landscape, 
# then reducing it when falling into an optimal minima troughs
early_stop = EarlyStopping(monitor='loss', patience=5)
callbacks = [reduce_lr, early_stop]

# train model
if args.tune:
    func = importlib.import_module('archs.' + args.arch).get_meta_model(input_shape)
    model = tune(func, X_train, Y_train, X_test, Y_test, \
                 batch_size=args.batch, epochs=50, max_epochs=args.epochs)
    model.build(input_shape=input_shape)
else:
    model.fit(X_train, Y_train, batch_size=args.batch, epochs=args.epochs, \
              callbacks=callbacks, validation_data=([X_test], [Y_test]))

# evaluate the model
loss = model.evaluate(X_test, Y_test)
print('Loss:', loss)

# make a prediction
#prediction = model.predict(X_test)
#print('Prediction:', prediction)

# save model
model.save(f"models/{args.arch}/1")
np.savez(os.path.join(SNPDIR, "test.npz"), X_test=X_test, Y_test=Y_test, X_train=X_train, Y_train=Y_train)
