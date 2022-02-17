import json
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from RateStateModel import RateStateModel
import numpy as np
from MCMC import MCMC
import sys
import time
import matplotlib.pyplot as plt

import generate_dataset


def main():

    model = RateStateModel(dc=1.0) # forward model
    t,acc = model.evaluate() # true data
    acc_noise = acc+1.0*np.abs(acc)*np.random.randn(acc.shape[0],acc.shape[1]) #synthetic data

    #----------------------------------------------------------------------------------------------------------------
    # Prepare data
    y_ = acc_noise.reshape(-1, 1)
    t_ = t.reshape(-1, 1)
    num_ = 10
    t_test, y_test, t_train, y_train  = generate_dataset.train_test_split(t_, y_, num_)  

    NN_layers=[30]*3
    model_NN= Sequential()
    model_NN.add(Dense(NN_layers[0], input_dim=1, kernel_initializer='random_uniform', activation='relu'))

    for ilayer in range(1,len(NN_layers)):
        model_NN.add(Dense(NN_layers[ilayer],kernel_initializer='random_uniform', activation='relu'))

    model_NN.add(Dense(1, activation='linear'))
    model_NN.compile(loss='mse', optimizer='adam')
    model_NN.summary()

    history = model_NN.fit(t_train, y_train,epochs=200,batch_size=50,verbose=1,validation_split=0.1)

    yval_pred=model_NN.predict(t_train)

    plt.figure()
    plt.plot(t_train,y_train,'-o')
    plt.plot(t_train,yval_pred)
    plt.show()


    yval_pred=model_NN.predict(t_test)

    plt.figure()
    plt.plot(t_test,y_test,'-o')
    plt.plot(t_test,yval_pred)
    plt.show()

    print(t_train.shape, y_train.shape, t_test.shape, y_test.shape)
 

if __name__ == '__main__':

    main()




