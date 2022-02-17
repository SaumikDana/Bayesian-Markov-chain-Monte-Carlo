# Author: Saumik Dana

'''

Example of using a LSTM encoder-decoder to model a synthetic time series 

'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

import generate_dataset
import lstm_encoder_decoder
import plotting 
import plotting_again 

matplotlib.rcParams.update({'font.size': 10})

#----------------------------------------------------------------------------------------------------------------
# generate dataset for LSTM

#dc_ = 1000
#t, y_solution, y = generate_dataset.synthetic_data(dc_) # time series and y time series

sed = np.loadtxt('../elcentro_data/elcentro.dat', unpack = True)
t = sed[0,:]
y = sed[1,:]

#----------------------------------------------------------------------------------------------------------------
# window dataset

# set size of input/output windows 
iw = 5
ow = 5 
s = 10 # stride

# generate windowed training/test datasets
# for each window, there is an input window and output window
# data in input window is used to train the model, data in output window is compared agaianst the model prediction
# So we are predicting only for the portion in the output window !!!

y_train = y.reshape(-1, 1)
t_train = t.reshape(-1, 1)

num_samples_train,Ttrain, Xtrain, Ytrain= generate_dataset.windowed_dataset(t_train, y_train, input_window = iw, output_window = ow, stride = s) 

#----------------------------------------------------------------------------------------------------------------
# LSTM encoder-decoder

# convert windowed data from np.array to PyTorch tensor
T_train, X_train, Y_train = generate_dataset.numpy_to_torch_again(Ttrain, Xtrain, Ytrain)

# specify model parameters and train
hidden_size = 10
batch_size = 10
n_epochs = 100
num_layers = 1 

model = lstm_encoder_decoder.lstm_seq2seq(input_size = X_train.shape[2], hidden_size = hidden_size, num_layers = num_layers) #X_train.shape[2] is 1
loss = model.train_model(X_train, Y_train, n_epochs = n_epochs, target_len = iw+ow, batch_size = 1, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)
plotting_again.plot_train_test_results(model, Ttrain, Xtrain, Ytrain, num_samples_train)

plt.close('all')

