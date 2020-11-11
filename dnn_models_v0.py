from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import optimizers
from keras import regularizers

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    
    if model_tag == "model_sgd":
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

    if model_tag == "model_adam":
        model.add(Dense(20, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
               
        model.add(Dense(50, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        model.add(Dense(70))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        model.add(Dense(50))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        model.add(Dense(20))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        return model