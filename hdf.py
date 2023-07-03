""" hfd - Hafen Düsen Fog - hard (to model) function synthetic data generator """
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split


def create_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(1,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=['mae'])

    return model

def main(args):
    num_samples = args.num_samples
    noise_level = 0.1
    xmin, xmax = -10, 10

    if args.function == 'heaviside':
        x = np.linspace(xmin, xmax, num_samples)
        y = np.heaviside(x, 1)
        
    elif args.function == 'periodic_var_freq': # period functions with variable frequency
        x = np.linspace(xmin, xmax, num_samples)
        y = np.sin(x * x)

    elif args.function == 'high_dim_interactions':
        x = np.linspace(xmin, xmax, num_samples)
        y = np.linspace(xmin, xmax, num_samples)
        z = np.linspace(xmin, xmax, num_samples)
        y = x**3 + y**2*z + y*z**2

    elif args.function == 'dirichlet':
        # note: the dirichlet function is difficult to plot.
        # see https://mathworld.wolfram.com/DirichletFunction.html
        x = np.linspace(xmin, xmax, num_samples)
        fractional, integral = np.modf(x)
        y = np.where(np.abs(fractional) < 1e-9, 1, 0)

    elif args.function == 'non_stationary':
        x = np.linspace(xmin, xmax, num_samples)
        mean, var = np.sin(2 * np.pi * x), np.exp(2 * x)
        y = np.random.normal(loc=mean, scale=np.sqrt(var))

    elif args.function == 'noisy_func':
        x = np.linspace(xmin, xmax, num_samples)
        y = noisy_func(x, noise_level)

    else:
        raise ValueError(f"Unknown function: {args.function}")

    if args.noise > 0:
        y += np.random.normal(scale=args.noise, size=num_samples)

    x = x[:, np.newaxis] # Add an extra dimension for the neural network
    y = y[:, np.newaxis] # Add an extra dimension for the neural network
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = create_model()
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=32, validation_data=(x_test, y_test))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, s=2)
    plt.title("Train data and model prediction")
    plt.scatter(x_train, model.predict(x_train), color='r', s=1)
    #plt.ylim(ymin, ymax)

    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, s=1)
    plt.title("Test data and model prediction")
    plt.scatter(x_test, model.predict(x_test), color='r', s=1)
    #plt.ylim(ymin, ymax)
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", "-f", type=str, default='heaviside', help="function for which to generate synthetic data")
    parser.add_argument("--num_samples", "-n", type=int, required=True, help="number of samples")
    parser.add_argument("--batch", "-b", type=int, default=10, help="number of epochs")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="number of epochs")
    parser.add_argument("--noise", type=float, default=0, help="amount of noise to add [0, 1]")
    args = parser.parse_args()
    
    main(args)

