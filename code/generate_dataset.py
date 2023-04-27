# Author: Saumik Dana

'''

Generate a synthetic dataset for our LSTM encoder-decoder
We will consider a noisy sinusoidal curve 

'''

import numpy as np
import torch

def train_test_split(t, y, num_):
  '''
  
  split time series into train/test sets
  
  : param t:                      time array
  : para y:                       feature array
  : para split:                   percent of data to include in training set 
  : return t_train, y_train:      time/feature training and test sets;  
  :        t_test, y_test:        (shape: [# samples, 1])
  
  '''

  count1 = 0; count2 = 0
  for indx in range(0, y.shape[0]):
      if indx % num_: # if the remainder on division by num_ is not zero
         count1 += 1
      else :
         count2 += 1
  
  t_train,y_train,t_test,y_test = np.zeros((count1,2)),np.zeros((count1,2)),np.zeros((count2,2)),np.zeros((count2,2))

  count1 = 0; count2 = 0
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
 
    L                           = y.shape[0]
    num_samples                 = (L - window) // stride + 1
    Y                           = np.zeros([window, num_samples, num_features])     
    T                           = np.zeros([window, num_samples, num_features])     
   
    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x             = stride * ii
            end_x               = start_x + window
            index               = range(start_x,end_x)
            Y[0:window, ii, ff] = y[index, ff] 
            T[0:window, ii, ff] = t[index, ff]

    return num_samples, T, Y


def numpy_to_torch(Ttrain, Ytrain):
    '''
    convert numpy array to PyTorch tensor
    
    '''
    T_train_torch = torch.from_numpy(Ttrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)
    
    return T_train_torch, Y_train_torch
