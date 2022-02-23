# Author: Saumik Dana

import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_train_test_results(lstm_model, T_, X_, Y_, stride, window, dataset_type, objective, num_samples, num_dc, dc_, num_tsteps):
  '''
  plot examples of the lstm encoder-decoder evaluated on the training/test data
  
  '''

  Y_return = np.zeros([int(num_samples*window)])     
  count_dc = 0
  for dc in dc_:

      X = np.zeros([int(num_samples*window/num_dc)]) 
      Y = np.zeros([int(num_samples*window/num_dc)])     
      T = np.zeros([int(num_samples*window/num_dc)])     

      num_samples_per_dc = int(num_samples/num_dc) 

      for ii in range(num_samples_per_dc):

          start = ii*stride
          end = start + window

          train_plt = X_[:, count_dc*num_samples_per_dc+ii, :]
          Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len = window)

          X[start:end] = Y_[:, count_dc*num_samples_per_dc+ii, 0]
          Y[start:end] = Y_train_pred[:, 0]
          T[start:end] = T_[:, count_dc*num_samples_per_dc+ii, 0]
          Y_return[count_dc*num_tsteps+start:count_dc*num_tsteps+end] = Y_train_pred[:, 0]

      plt.figure()
      plt.plot(T, X, '-', color = (0.2, 0.42, 0.72), linewidth = 1.0, markersize = 1.0, label = 'Target')
      plt.plot(T, Y, '-', color = (0.76, 0.01, 0.01), linewidth = 1.0, markersize = 1.0, label = '%s' % objective)
      plt.xlabel('$Time (sec)$')
      plt.ylabel('$a (\mu m/s^2)$')
      plt.legend(frameon=False)
      plt.suptitle('%s data set for dc=%s $\mu m$' % (dataset_type,dc), x = 0.445, y = 1.)
      plt.tight_layout()
      plt.savefig('plots/%s_%s.png' % (dc,dataset_type))

      count_dc += 1

      del X,Y,T


def rom_evaluate(self, dc):
     
    num_steps = int(np.floor((self.consts["t_final"]-self.consts["t_start"])/self.consts["delta_t"]))
    window = self.consts["window"]

    # Create arrays to store trajectory
    t = np.zeros((num_steps,1)) 
    acc = np.zeros((num_steps,1)) 

    t[0,0] = self.consts["t_start"]

    k = 1
    while k < num_steps:
        t[k,0] = t[k-1,0] + self.consts["delta_t"]
        k += 1
 
    num_steps_ = int(num_steps/window) 
    train_plt = np.zeros((num_steps_,2)) 
    
    for ii in range(num_steps_):

        start = ii*stride
        end = start + window

        train_plt[0:window,0] = t[start:end, 0]
        train_plt[0:window,1] = dc
        
        Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len = window)
        acc[start:end,0] = Y_train_pred[:, 0]
    
    return t.T, acc.T


def plot_results(lstm_model, T_, X_, Y_, stride, window, dataset_type, objective, num_samples, num_q, q_, num_tsteps):
  '''
  plot examples of the lstm encoder-decoder evaluated on the training/test data
  
  '''

  Y_return = np.zeros([int(num_samples*window)])     
  count_q = 0
  for q in q_:

      X = np.zeros([int(num_samples*window/num_q)]) 
      Y = np.zeros([int(num_samples*window/num_q)])     
      T = np.zeros([int(num_samples*window/num_q)])     

      num_samples_per_q = int(num_samples/num_q) 

      for ii in range(num_samples_per_q):

          start = ii*stride
          end = start + window

          train_plt = X_[:, count_q*num_samples_per_q+ii, :]
          Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len = window)

          X[start:end] = Y_[:, count_q*num_samples_per_q+ii, 0]
          Y[start:end] = Y_train_pred[:, 0]
          T[start:end] = T_[:, count_q*num_samples_per_q+ii, 0]
          Y_return[count_q*num_tsteps+start:count_q*num_tsteps+end] = Y_train_pred[:, 0]

      plt.figure()
      plt.plot(T, X, '-', color = (0.2, 0.42, 0.72), linewidth = 1.0, markersize = 1.0, label = 'Target')
      plt.plot(T, Y, '-', color = (0.76, 0.01, 0.01), linewidth = 1.0, markersize = 1.0, label = '%s' % objective)
      plt.xlabel('Time stamp')
      plt.ylabel('Disp X $(m)$')
      plt.legend(frameon=False)
      plt.suptitle('%s data set for q=%s $\mu m$' % (dataset_type,q), x = 0.445, y = 1.)
      plt.tight_layout()
      plt.savefig('plots/%s_%s.png' % (q,dataset_type))

      count_q += 1

      del X,Y,T


