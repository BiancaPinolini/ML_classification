from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import regularizers, optimizers



def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()    
    if model_tag == "2l_50n_relu":
        model.add(Dense(50, input_dim=input_dim, activation="relu"))
        model.add(Dense(50, activation="relu"))        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "2l_64n_batchnorm_relu":
        model.add(Dense(64, input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        
        model.add(Dense(64))    
        model.add(BatchNormalization())  
        model.add(Activation("relu"))
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "2l_64n_l2_05_batchnorm_relu":
        model.add(Dense(64, input_dim=input_dim, kernel_regularizer=regularizers.l2(0.05)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        
        model.add(Dense(64, kernel_regularizer=regularizers.l2(0.05)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "2l_30n_dropout5_l2_batchnorm_relu":
        model.add(Dense(30, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(30, activation="relu",kernel_regularizer=regularizers.l2(0.01)))    
        model.add(BatchNormalization())   
        model.add(Dropout(0.5))       
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "4l_triangle_l2_batchnorm_relu":
        model.add(Dense(128, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(16,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    
    if model_tag == "5l_20_70n_dropout01_l2_batchnorm_relu":
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
        
        return model