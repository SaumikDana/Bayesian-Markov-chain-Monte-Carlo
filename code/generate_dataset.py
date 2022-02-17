# Author: Laura Kulowski

'''

Generate a synthetic dataset for our LSTM encoder-decoder
We will consider a noisy sinusoidal curve 

'''

from RateStateModel import RateStateModel # RSF model
import numpy as np
import torch
import math

def synthetic_data(dc = 1.0):
    
    '''
    create synthetic time series dataset
    : param Nt:       number of time steps 
    : param tf:       final time
    : return t, y:    time, feature arrays
    '''
    ### RSF Model ###
    consts={}
    consts["a"]=0.011;  consts["b"]=0.014;   consts["mu_ref"]=0.6
    consts["V_ref"]= 1;   consts["k1"]= 1e-7
    # Time range
    consts["t_start"]=0.0;  consts["t_final"]=50.0; consts["delta_t"]=5e-2
    # Initial conditions
    consts["mu_t_zero"] = 0.6;  consts["V_ref"] = 1.0

    consts["Dc"] = dc

    consts["mu_ref"] = 0.6
    consts["RadiationDamping"]=True    
    model=RateStateModel(consts,False) # forward model

    t_,acc = model.evaluate(consts) # true data
    acc_noise = acc+1.0*np.abs(acc)*np.random.randn(acc.shape[0],acc.shape[1]) #synthetic data
    ### RSF Model ###

    return t_.T.flatten(), acc.T.flatten(), acc_noise.T.flatten()

def train_test_split(t, y, num_dc, num_):
  '''
  
  split time series into train/test sets
  
  : param t:                      time array
  : para y:                       feature array
  : para split:                   percent of data to include in training set 
  : return t_train, y_train:      time/feature training and test sets;  
  :        t_test, y_test:        (shape: [# samples, 1])
  
  '''

  count1 = 0; count2 = 0;
  for indx in range(0, y.shape[0]):
      if indx % num_: # if the remainder on division by num_ is not zero
         count1 += 1
      else :
         count2 += 1
  
  t_train,y_train,t_test,y_test = np.zeros((count1,2)),np.zeros((count1,2)),np.zeros((count2,2)),np.zeros((count2,2))

  count1 = 0; count2 = 0;
  for indx in range(0, y.shape[0]):
      if indx % num_: # if the remainder on division by num_ is not zero
         t_train[count1,0] = t[indx,0]
         y_train[count1,0] = y[indx,0]
         t_train[count1,1] = t[indx,1]
         y_train[count1,1] = y[indx,1]
         count1 += 1
      else :
         t_test[count2,0] = t[indx,0]
         y_test[count2,0] = y[indx,0]
         t_test[count2,1] = t[indx,1]
         y_test[count2,1] = y[indx,1]
         count2 += 1
  
  return t_train, y_train, t_test, y_test 


def windowed_dataset(t, y, window, stride, num_features = 1):
    '''
    create a windowed dataset
    
    : param y:                time series feature (array)
    : param input_window:     number of y samples to give model 
    : param output_window:    number of future y samples to predict  
    : param stride:            spacing between windows   
    : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''
 
    L = y.shape[0]
    num_samples = int(math.ceil(L/stride))

    Y = np.zeros([window, num_samples, num_features])     
    T = np.zeros([window, num_samples, num_features])     
   
    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + window
            Y[0:window, ii, ff] = y[start_x:end_x, ff] 
            T[0:window, ii, ff] = t[start_x:end_x, ff]

    return num_samples, T, Y


def numpy_to_torch(Ttrain, Ytrain):
    '''
    convert numpy array to PyTorch tensor
    
    '''
    T_train_torch = torch.from_numpy(Ttrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)
    
    return T_train_torch, Y_train_torch
