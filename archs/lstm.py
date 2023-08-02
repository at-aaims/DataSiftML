from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape, Flatten
from . import MetaModel


def build_model(input_shape, units=256, activation='elu', dropout=0.1, lr=3e-4):

    input_layer = Input(shape=input_shape, name='inputs')
    x = LSTM(units, activation=activation, return_sequences=True)(input_layer)
    x = LSTM(units, activation=activation, return_sequences=True)(x)
    x = LSTM(units, activation=activation, return_sequences=True)(x)
    x = LSTM(units, activation=activation, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(units, activation=activation)(x)
    x = Dense(units//2, activation=activation)(x)
    x = Dense(3, activation='linear')(x)
    outputs = Reshape((3, 1), name='outputs')(x)

    model = Model(inputs=[input_layer], outputs=[outputs])
    model.compile(loss='mae', optimizer='adam')

    return model


get_meta_model = MetaModel(build_model).get_meta_model
