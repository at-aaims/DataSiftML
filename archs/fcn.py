from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_model(input_shape, units1=2000, units2=2000, activation='elu', window=1):
    activation = 'elu'
    units0 = int(input_shape[-1])
    print('Layer 0 units:', units0)
    input_layer = Input(shape=input_shape, name='inputs')
    
    x = Dense(units0, activation=activation)(input_layer)
    x = Dense(int(units1), activation=activation)(x)
    x = Dense(int(units2), activation=activation)(x)
    out = Dense(1, activation='linear', name='outputs')(x)
    model = Model(inputs=[input_layer], outputs=[out])
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mae', optimizer=optimizer)

    return model


def get_meta_model(input_shape):

    def meta_model(hp):
        units1 = hp.Int('units1', min_value=2, max_value=8000, step=800)
        units2 = hp.Int('units2', min_value=2, max_value=8000, step=800)
        activation = hp.Choice('activation', ['elu', 'gelu', 'selu'])
        dropout = hp.Float('dropout_rate', 0.0, 0.5, sampling='linear')
        hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
        return build_model(input_shape, units1, units2)

    return meta_model
