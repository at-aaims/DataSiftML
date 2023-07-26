import kerastuner as kt

def scale(func, x):
    """convert data to 2D scale and reshape back to 3D"""
    return func(x.reshape(-1, x.shape[-1])).reshape(x.shape)

def tune(func, x_train, y_train, x_test, y_test, batch_size,
         objective='val_loss', epochs=5, max_epochs=2):
    tuner = kt.Hyperband(func, objective=objective, max_epochs=max_epochs, overwrite=True)
    tuner.search(x_train, y_train, batch_size=batch_size, epochs=epochs,
                 validation_data=(x_test, y_test))
    print(tuner.results_summary(1))
    return tuner.get_best_models()[0]
