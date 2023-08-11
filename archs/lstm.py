from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


def build_model(input_shape, units=288, activation='elu', dropout=0.5, lr=0.0003, window=3):

    input_layer = Input(shape=input_shape, name='inputs')
    x = LSTM(units, activation=activation, return_sequences=True)(input_layer)
    x = LSTM(units, activation=activation, return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(units, activation=activation)(x)
    x = Dense(units//2, activation=activation)(x)
    x = Dense(window, activation='linear')(x)
    outputs = Reshape((window, 1), name='outputs')(x)

    model = Model(inputs=[input_layer], outputs=[outputs])
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='mae', optimizer=optimizer)

    return model

#from . import MetaModel
#get_meta_model = MetaModel(build_model).get_meta_model

def get_meta_model(input_shape):

    def meta_model(hp):
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        #activation = hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu', 'gelu', 'tanh', 'selu'])
        activation = hp.Choice('activation', ['elu', 'gelu', 'selu'])
        dropout = hp.Float('dropout_rate', 0.0, 0.5, sampling='linear')
        lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
        return build_model(input_shape, units, activation, dropout, lr)

    return meta_model
