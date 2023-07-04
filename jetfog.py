""" 
 Hard-to-model function synthetic data generator

 Usage examples:

    python jetfog.py -f heaviside -n 100 -e 100 --plot --spacing gaussian

"""
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

def predict_with_dropout(model, x, T=100):
    """ function to make MC Dropout predictions """
    Y_T = np.array([model(x, training=True) for _ in range(T)])
    mean = Y_T.mean(axis=0)
    variance = Y_T.var(axis=0)
    return mean, variance

def main(args):
    num_samples = args.num_samples
    noise_level = 0.1
    xmin, xmax = -10, 10

    # Generate x-spacing
    if args.spacing == 'sine': 
        x = np.linspace(0, 180, num_samples)
        x = np.sin(np.radians(x))

    elif args.spacing == 'gaussian': 
        x = np.random.normal(loc=0, scale=2, size=num_samples)

    else: # linear
        x = np.linspace(xmin, xmax, num_samples)

    # Generate data from functions
    if args.function == 'heaviside':
        y = np.heaviside(x, 1) + 1
        
    elif args.function == 'periodic_var_freq': # period functions with variable frequency
        y = np.sin(x * x)

    elif args.function == 'high_dim_interactions':
        y = z = x
        y = x**3 + y**2*z + y*z**2

    elif args.function == 'dirichlet':
        # note: the dirichlet function is difficult to plot.
        # see https://mathworld.wolfram.com/DirichletFunction.html
        fractional, integral = np.modf(x)
        y = np.where(np.abs(fractional) < 1e-9, 1, 0)

    elif args.function == 'non_stationary':
        mean, var = np.sin(2 * np.pi * x), np.exp(2 * x)
        y = np.random.normal(loc=mean, scale=np.sqrt(var))

    else:
        raise ValueError(f"Unknown function: {args.function}")

    if args.noise > 0:
        y += np.random.normal(scale=args.noise, size=num_samples)

    x = x[:, np.newaxis] # Add an extra dimension for the neural network
    y = y[:, np.newaxis] # Add an extra dimension for the neural network
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = create_model()
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch, validation_data=(x_test, y_test))

    # Make predictions
    y_predict_train = model.predict(x_train)
    mean_train, var_train = predict_with_dropout(model, x_train)
    stddev_train = np.sqrt(var_train)
    max_stddev_train = np.max(stddev_train)
    print('max_stddev_train:', max_stddev_train)
    stddev_train /= max_stddev_train

    y_predict_test = model.predict(x_test)
    mean_test, var_test = predict_with_dropout(model, x_test)
    stddev_test = np.sqrt(var_test)
    max_stddev_test = np.mean(stddev_test)
    print('max_stddev_test:', max_stddev_test)
    stddev_test /= max_stddev_test

    mean_train = np.squeeze(mean_train)
    stddev_train = np.squeeze(stddev_train)
    mean_test = np.squeeze(mean_test)
    stddev_test = np.squeeze(stddev_test)

    print(stddev_train)
    print(stddev_test)

    if args.plot:

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(x_train, y_train, s=2)
        plt.title("Train data and model prediction")
        plt.scatter(x_train, model.predict(x_train), color='r', s=1)
        plt.errorbar(x_train, mean_train, yerr=stddev_train, fmt='o', color='r', \
                     ecolor='gray', elinewidth=1, capsize=1, markersize=2)
        for x in x_train: plt.axvline(x=x, color='r', linestyle=':', alpha=0.1)

        plt.subplot(1, 2, 2)
        plt.scatter(x_test, y_test, s=1)
        plt.title("Test data and model prediction")
        plt.scatter(x_test, model.predict(x_test), color='r', s=1)
        plt.errorbar(x_test, mean_test, yerr=stddev_test, fmt='o', color='r', \
                     ecolor='gray', elinewidth=1, capsize=1, markersize=2)
        
        plt.show()


if __name__ == '__main__':

    funcs = ['heaviside', 'periodic_var_freq', 'dirichlet', 'non_stationary']
    spacing = ['uniform', 'sine', 'gaussian']

    parser = argparse.ArgumentParser()
    parser.add_argument("--function", "-f", type=str, default='heaviside', choices=funcs, \
                        help="function for which to generate synthetic data")
    parser.add_argument("--num_samples", "-n", type=int, required=True, help="number of samples")
    parser.add_argument("--batch", "-b", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="number of epochs")
    parser.add_argument("--noise", type=float, default=0, help="amount of noise to add [0, 1]")
    parser.add_argument("--plot", action='store_true', help="plot results")
    parser.add_argument("--spacing", choices=spacing, help="type of spacing to use")
    args = parser.parse_args()
    
    main(args)

