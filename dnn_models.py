# from keras.models import Sequential, load_model
# from keras.activations import relu
# from keras.layers import Dense, LeakyReLU, Dropout
# from keras.optimizers import Adam, SGD
# from keras import metrics

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import optimizers
from keras import regularizers

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    
    if model_tag == "my_model":
        model.add(Dense(30, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Dense(50, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(30))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='sgd',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
    return model# from keras.models import Sequential, load_model
# from keras.activations import relu
# from keras.layers import Dense, LeakyReLU, Dropout
# from keras.optimizers import Adam, SGD
# from keras import metrics

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import optimizers
from keras import regularizers

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    
    if model_tag == "my_model":
        model.add(Dense(30, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Dense(50, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(30))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
    return model# from keras.models import Sequential, load_model
# from keras.activations import relu
# from keras.layers import Dense, LeakyReLU, Dropout
# from keras.optimizers import Adam, SGD
# from keras import metrics

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import optimizers
from keras import regularizers

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    
    if model_tag == "my_model":
        model.add(Dense(30, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Dense(50, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(30))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
    return model# from keras.models import Sequential, load_model
# from keras.activations import relu
# from keras.layers import Dense, LeakyReLU, Dropout
# from keras.optimizers import Adam, SGD
# from keras import metrics

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import optimizers
from keras import regularizers

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    
    if model_tag == "my_model":
        model.add(Dense(30, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Dense(50, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(30))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='sgd',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
    return model# from keras.models import Sequential, load_model
# from keras.activations import relu
# from keras.layers import Dense, LeakyReLU, Dropout
# from keras.optimizers import Adam, SGD
# from keras import metrics

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import optimizers
from keras import regularizers

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    
    if model_tag == "my_model":
        model.add(Dense(30, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Dense(50, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(30))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='sgd',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
    return model# from keras.models import Sequential, load_model
# from keras.activations import relu
# from keras.layers import Dense, LeakyReLU, Dropout
# from keras.optimizers import Adam, SGD
# from keras import metrics

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import optimizers
from keras import regularizers

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    
    if model_tag == "my_model":
        model.add(Dense(30, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Dense(50, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(30))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='sgd',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
    return model
# from keras.models import Sequential, load_model
# from keras.activations import relu
# from keras.layers import Dense, LeakyReLU, Dropout
# from keras.optimizers import Adam, SGD
# from keras import metrics

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import optimizers
from keras import regularizers

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    
    if model_tag == "my_model":
        model.add(Dense(30, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Dense(50, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(30))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
    return model
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   