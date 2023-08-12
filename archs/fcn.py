from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, activation='elu',dropout=0.20, lr=0.0001, window=3):
    
    units = int(input_shape[-1])
    print('Layer 0 units:', units)
    input_layer = Input(shape=input_shape, name='inputs')
    
    x = Dense(units, activation=activation)(input_layer)
    x = Dense(int(units//2), activation=activation)(x)
    #x = Dropout(0.15)(x)
    x = Dense(int(units//4), activation=activation)(x)
    x = Dropout(0.25)(x)
    x = Dense(int(units//8), activation=activation)(x)
    #x = Dropout(0.15)(x)
    x = Dense(int(units//16), activation=activation)(x)
    #x = Dense(int(units//16), activation=activation)(x)

    out = Dense(1, activation='linear', name='outputs')(x)
    model = Model(inputs=[input_layer], outputs=[out])
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='mae', optimizer=optimizer)

    return model


