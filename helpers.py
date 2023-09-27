import numpy as np
import kerastuner as kt

def scale(func, x):
    """convert data to 2D scale and reshape back to 3D"""
    return func(x.reshape(-1, x.shape[-1])).reshape(x.shape)

def tune(func, x_train, y_train, x_test, y_test, batch_size,
         objective='val_loss', epochs=50, max_epochs=100):
    tuner = kt.Hyperband(func, objective=objective, max_epochs=max_epochs, overwrite=True)
    tuner.search(x_train, y_train, batch_size=batch_size, epochs=epochs,
                 validation_data=(x_test, y_test))
    print(tuner.results_summary(1))
    return tuner.get_best_models()[0]

def scale_probabilities(probs, a=0.01, b=0.99):
    """
    Scale a list of probabilities linearly from range [a, b].
    
    Args:
    - probs: List of probabilities
    - a, b: Range for scaling (default is [0.01, 0.99])

    Returns:
    - Scaled list of probabilities
    """
    A, B = min(probs), max(probs)
    scaled_probs = [(x - A) * (b - a) / (B - A) + a for x in probs]
    return np.array(scaled_probs)

def print_stats(label, X, Y):

    stats = lambda x : f"min: {np.amin(x):.04f}, mean: {np.mean(x):.04f}, max: {np.amax(x):.04f}"

    print(label)
    print('X[0]:', stats(X[:, 0]))
    print('X[1]:', stats(X[:, 1]))
    print('Y:', stats(Y[:]))

def verbose_io(func):
    def wrapper(*args, **kwargs):
        print(f"{func.__name__} {args[0]}")
        return func(*args, **kwargs)
    return wrapper

@verbose_io
def load(*args, **kwargs):
    return np.load(*args, **kwargs)

@verbose_io
def savez(*args, **kwargs):
    np.save(*args, **kwargs)

