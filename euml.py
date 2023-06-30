import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Define the functions
def heaviside(x, p):
    return tf.cast(x - p > 0, dtype=tf.float32)

def periodic_var_freq(x):
    return tf.math.sin(x * x)

def high_dim_interactions(x, y, z):
    return x**3 + y**2*z + y*z**2

def dirichlet(x):
    return tf.where(x == tf.floor(x), tf.ones_like(x), tf.zeros_like(x))

def non_stationary(x):
    return tf.where(x < 0, tf.random.normal(shape=x.shape), tf.ones_like(x))

def noisy_func(x, noise_level=0.1):
    return x + noise_level * tf.random.normal(shape=x.shape)

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

    if args.function == 'heaviside':
        x = tf.linspace(-10.0, 10.0, num_samples)
        p = tf.linspace(-10.0, 10.0, num_samples)
        y = heaviside(x, p)
        
    elif args.function == 'periodic_var_freq':
        x = tf.linspace(-10.0, 10.0, num_samples)
        y = periodic_var_freq(x)

    elif args.function == 'high_dim_interactions':
        x = tf.linspace(-10.0, 10.0, num_samples)
        y = tf.linspace(-10.0, 10.0, num_samples)
        z = tf.linspace(-10.0, 10.0, num_samples)
        y = high_dim_interactions(x, y, z)

    elif args.function == 'dirichlet':
        x = tf.linspace(-10.0, 10.0, num_samples)
        y = dirichlet(x)

    elif args.function == 'non_stationary':
        x = tf.linspace(-10.0, 10.0, num_samples)
        y = non_stationary(x)

    elif args.function == 'noisy_func':
        x = tf.linspace(-10.0, 10.0, num_samples)
        y = noisy_func(x, noise_level)

    else:
        raise ValueError(f"Unknown function: {args.function}")

    x = x[:, np.newaxis] # Add an extra dimension for the neural network
    y = y[:, np.newaxis] # Add an extra dimension for the neural network
    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=0.2, random_state=42)

    model = create_model()
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, s=1)
    plt.title("Train data and model prediction")
    plt.plot(x_train, model.predict(x_train), 'r', linewidth=2)

    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, s=1)
    plt.title("Test data and model prediction")
    plt.plot(x_test, model.predict(x_test), 'r', linewidth=2)
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", type=str, default='noisy_func', help="The function for which to generate synthetic data")
    parser.add_argument("--num_samples", "-n", type=int, help="number of samples")
    args = parser.parse_args()
    
    main(args)

