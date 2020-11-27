from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import regularizers, optimizers

def get_model(model_tag, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    
    if model_tag == "3l_150n_l2_batchnorm_relu":
        model.add(Dense(150, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(150,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(150,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    
    
    
    
    
    
    
    
    
    if model_tag == "2l_10n_relu":
        model.add(Dense(10, input_dim=input_dim, activation="relu"))
        model.add(Dense(10, activation="relu"))        
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "2l_50n_relu":
        model.add(Dense(50, input_dim=input_dim, activation="relu"))
        model.add(Dense(50, activation="relu"))        
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "2l_50n_batchnorm_relu":
        model.add(Dense(50, input_dim=input_dim, activation="relu"))
        model.add(BatchNormalization())
        
        model.add(Dense(50, activation="relu"))    
        model.add(BatchNormalization())   
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "2l_50n_l2_batchnorm_relu":
        model.add(Dense(50, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        
        model.add(Dense(50, activation="relu",kernel_regularizer=regularizers.l2(0.01)))    
        model.add(BatchNormalization())   
        
        model.add(Dense(1, activation="sigmoid"))
        return model

    if model_tag == "2l_30n_dropout05_l2_batchnorm_relu":
        model.add(Dense(30, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(30, activation="relu",kernel_regularizer=regularizers.l2(0.01)))    
        model.add(BatchNormalization())   
        model.add(Dropout(0.5))
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "2l_50n_dropout05_l2_batchnorm_relu":
        model.add(Dense(50, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(50, activation="relu",kernel_regularizer=regularizers.l2(0.01)))    
        model.add(BatchNormalization())   
        model.add(Dropout(0.5))
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "3l_100n_l2_batchnorm_relu":
        model.add(Dense(100, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(100, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(100, activation="relu"))    
        model.add(BatchNormalization())    
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "3l_100n_dropout005_relu":
        model.add(Dense(100, input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.05))
        
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.05))
        
        model.add(Dense(100, activation="relu"))   
        model.add(Dropout(0.05))     
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model

    if model_tag == "3l_50n_dropout05_l2_batchnorm_relu":
        model.add(Dense(50, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(50, activation="relu",kernel_regularizer=regularizers.l2(0.01)))    
        model.add(BatchNormalization())   
        model.add(Dropout(0.5))
        
        model.add(Dense(50, activation="relu"))    
        model.add(BatchNormalization())   
        model.add(Dropout(0.5))
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "3l_256n_dropout05_l2_batchnorm_relu":
        model.add(Dense(60, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(100, activation="relu",kernel_regularizer=regularizers.l2(0.01)))    
        model.add(BatchNormalization())   
        model.add(Dropout(0.5))
        
        model.add(Dense(256, activation="relu"))    
        model.add(BatchNormalization())   
        model.add(Dropout(0.5))
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "4l_50n_dropout01_l2_batchnorm_relu":
        model.add(Dense(50, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        model.add(Dense(50, activation="relu",kernel_regularizer=regularizers.l2(0.01)))    
        model.add(BatchNormalization())   
        model.add(Dropout(0.1))
        
        model.add(Dense(50, activation="relu"))    
        model.add(BatchNormalization())   
        model.add(Dropout(0.1))
        
        model.add(Dense(50, activation="relu"))    
        model.add(BatchNormalization())   
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "4l_50n_dropout01_l2_relu":
        model.add(Dense(80, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.1))
        
        model.add(Dense(50,activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.1))
        
        model.add(Dense(50,activation="relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(50,activation="relu"))
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "4l_100n_dropout02_l2_batchnorm_relu":
        model.add(Dense(150, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())   
        model.add(Dropout(0.2))
        
        model.add(Dense(100,activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())   
        model.add(Dropout(0.2))
        
        model.add(Dense(100,activation="relu"))
        model.add(BatchNormalization())   
        model.add(Dropout(0.2))
        
        model.add(Dense(100,activation="relu"))
        model.add(BatchNormalization())   
        model.add(Dropout(0.2))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    
    if model_tag == "4l_256n_dropout005_l2_relu":
        model.add(Dense(256, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.1))
        
        model.add(Dense(256,activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.05))
        
        model.add(Dense(256,activation="relu"))
        model.add(Dropout(0.05))
        
        model.add(Dense(256,activation="relu"))
        model.add(Dropout(0.05))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model

    if model_tag == "4l_256n_dropout001_l2_relu":
        model.add(Dense(256, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.01))
        
        model.add(Dense(256,activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.01))
        
        model.add(Dense(256,activation="relu"))
        model.add(Dropout(0.01))
        
        model.add(Dense(256,activation="relu"))
        model.add(Dropout(0.01))
        
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
    
    if model_tag == "4l_100n_l2_batchnorm_relu":
        model.add(Dense(100, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.01))
               
        model.add(Dense(100, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.01))
        
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.01))
        
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.01))
                  
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "5l_50n_dropout01_relu":
        model.add(Dense(50, input_dim=input_dim, activation="relu"))
        for i in range(4):
            model.add(Dropout(0.1))
            model.add(Dense(50, activation="relu"))   
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

    if model_tag == "5l_20_70n_dropout05_l2_batchnorm_relu":
        model.add(Dense(20, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
               
        model.add(Dense(50, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(70))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(50))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(20))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "5l_100_150n_dropout01_l2_batchnorm_relu":
        model.add(Dense(150, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
               
        model.add(Dense(150, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "5l_200n_l2_batchnorm_relu":
        model.add(Dense(200, input_dim=input_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.01))
               
        model.add(Dense(200, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.01))
        
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.01))
        
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.01))
        
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.01))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "5l_100n_dropout05_l2_batchnorm_relu":
        model.add(Dense(150, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(150,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "5l_200n_dropout05_l2_batchnorm_relu":
        model.add(Dense(200, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(200,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(200,activation="relu"))
        model.add(Dropout(0.4))
        
        model.add(Dense(200,activation="relu"))
        model.add(Dropout(0.4))
        
        model.add(Dense(200,activation="relu"))
        model.add(Dropout(0.4))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "5l_150n_dropout02_l2_batchnorm_relu":
        model.add(Dense(200, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        
        model.add(Dense(200,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        
        model.add(Dense(150,activation="relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(150,activation="relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(150,activation="relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "5l_150n_dropout02_l2_batchnorm_relu":
        model.add(Dense(200, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        
        model.add(Dense(200,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        
        model.add(Dense(150,activation="relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(150,activation="relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(150,activation="relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "5l_300n_dropout02_l2_batchnorm_relu":
        model.add(Dense(300, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        
        model.add(Dense(300,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        
        model.add(Dense(200,activation="relu"))
        model.add(Dropout(0.4))
        
        model.add(Dense(200,activation="relu"))
        model.add(Dropout(0.4))
        
        model.add(Dense(200,activation="relu"))
        model.add(Dropout(0.4))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
   
    if model_tag == "6l_70n_dropout05_l2_batchnorm_relu":
        model.add(Dense(100, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(100,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(80,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.5))
        
        model.add(Dense(70,activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(70,activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(70,activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "6l_100n_dropout05_l2_batchnorm_relu":
        model.add(Dense(150, input_dim=input_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(150,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(100,activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.5))
        
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "6l_256n_l2_batchnorm_tanh":
        model.add(Dense(256, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(64))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(64))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(64))
        model.add(Activation('tanh'))
        model.add(Dropout(0.3))
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "6l_256n_l2_batchnorm_relu":
        model.add(Dense(256, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(1, activation="sigmoid"))
        return model
    
    if model_tag == "8l_64n_dropout01_l2_batchnorm_relu":
        model.add(Dense(64, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.3))
        
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.3))
        
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.3))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.1))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.1))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.1))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.1))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.1)) 
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    if model_tag == "8l_64n_dropout005_l2_batchnorm_relu":
        model.add(Dense(64, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.2))
        
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.2))
        
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.2))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.1))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.05))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.05))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu")) 
        model.add(Dropout(0.05))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.05)) 
        
        model.add(Dense(1, activation="sigmoid"))
        
        return model