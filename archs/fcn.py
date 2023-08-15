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
    #x = Dropout(0.15)(x)
    x = Dense(int(units2), activation=activation)(x)
    #x = Dropout(0.25)(x)
    #x = Dense(int(units3), activation=activation)(x)
    #x = Dropout(0.15)(x)
    #x = Dense(int(units), activation=activation)(x)
    #x = Dense(int(units//16), activation=activation)(x)

    out = Dense(1, activation='linear', name='outputs')(x)
    model = Model(inputs=[input_layer], outputs=[out])
    optimizer = Adam(learning_rate=0.001)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.7, patience=2, min_lr=1e-8)
    # FCN networks can be highly noisy at initial training if lr is high.
    # High lr is often desired to explore most of the solution landscape, then reducing it when falling into an optimal minima troughs
    early_stop = EarlyStopping(monitor='loss', patience=5)
    callbacks = [early_stop,reduce_lr]
    model.compile(loss='mae', optimizer=optimizer)

    return model, callbacks


def get_meta_model(input_shape):

    def meta_model(hp):
        units1 = hp.Int('units1', min_value=2, max_value=8000, step=800)
        units2 = hp.Int('units2', min_value=2, max_value=8000, step=800)
        #units3 = hp.Int('units3', min_value=2, max_value=8000, step=800)
        #activation = hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu', 'gelu', 'tanh', 'selu'])
        activation = hp.Choice('activation',['elu'])  #hp.Choice('activation', ['elu', 'gelu', 'selu'])
        #dropout = hp.Float('dropout_rate'.0.10 #hp.Float('dropout_rate', 0.0, 0.5, sampling='linear')
        lr = 0.0001   #hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
        return build_model(input_shape, units1, units2)

    return meta_model
