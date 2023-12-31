class MetaModel:

    def __init__(self, func):
        self.func = func

    def get_meta_model(self, input_shape):

        def meta_model(hp):
            units = hp.Int('units', min_value=32, max_value=512, step=32)
            #activation = hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu', 'gelu', 'tanh', 'selu'])
            activation = hp.Choice('activation', ['elu', 'gelu', 'selu'])
            dropout = hp.Float('dropout_rate', 0.0, 0.5, sampling='linear')
            lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
            return self.func(input_shape, units, activation, dropout, lr)

        return meta_model
