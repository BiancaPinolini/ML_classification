from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import optimizers
from keras import regularizers
from keras.optimizers import Adam

def get_model(X, input_dim):
    
    num_dense_layers = X[0]
    num_input_nodes = X[1]
    num_dense_nodes = X[2]
    dropout_2_rate = X[3]
    
    print(X)

    print(">>> Creating model...")

    model = Sequential()
    model.add(Dense(num_input_nodes, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    

    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+1)
        model.add(Dense(num_dense_nodes, activation='relu', name=name))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_2_rate))
        
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model