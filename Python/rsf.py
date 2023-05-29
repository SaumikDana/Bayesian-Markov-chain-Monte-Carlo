from RateStateModel import RateStateModel
import numpy as np
from save_load import save_object, load_object
import lstm_encoder_decoder
from MCMC import MCMC
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class rsf:
   '''
   Driver class for RSF model
   '''
   def __init__(
      self, 
      number_slip_values=1, 
      lowest_slip_value=1.0, 
      largest_slip_value=1000.0, 
      plotfigs=False):
      
      # Define the range of values for the critical slip distance
      self.num_dc       = number_slip_values
      start_dc          = lowest_slip_value
      end_dc            = largest_slip_value
      self.dc_list      = np.linspace(start_dc,end_dc,self.num_dc)

      # Define file names for saving and loading data and LSTM model
      self.lstm_file    = 'model_lstm.json'
      self.data_file    = 'data.json'

      self.num_features = 2
      self.plotfigs     = plotfigs

      return
   
   def generate_time_series(self):

      # Create arrays to store the time and acceleration data for all values of dc
      entries = self.num_dc*self.model.num_tsteps
      acc_appended_noise = np.zeros(entries)  
      count_dc = 0

      for dc in self.dc_list:
         # Evaluate the model for the current value of dc
         self.model.Dc = dc
         t, acc, acc_noise = self.model.evaluate()
         self.plot_time_series(t, acc) # generate plots
         start = count_dc*self.model.num_tsteps
         end = start + self.model.num_tsteps
         acc_appended_noise[start:end] = acc_noise[:]
         # count_dc
         count_dc += 1
                  
      # Save the data using pickle
      save_object(acc_appended_noise,self.data_file)

      return

   def plot_time_series(self, t, acc):

      # Generate plots if desired
      if self.plotfigs:
         plt.figure()
         plt.title('$d_c$=' + str(self.model.Dc) + ' $\mu m$' + ' RSF solution')
         plt.plot(t[:], acc[:], linewidth=1.0,label='True')
         plt.xlim(self.model.t_start - 2.0, self.model.t_final)
         plt.xlabel('Time (sec)')
         plt.ylabel('Acceleration $(\mu m/s^2)$')
         plt.grid('on')
         plt.legend()        
      return
         
   def build_lstm(self, epochs=20, num_layers=1, batch_size=1):

      # build LSTM redeuced order model
      window  = int(self.model.num_tsteps/20)
      stride  = int(window/5)
      T       = self.t_appended.reshape(-1,self.num_features)
      data    = self.acc_appended.reshape(-1,self.num_features)
      n_train = (data.shape[0] - window) // stride + 1
      Ytrain  = np.zeros([window, n_train, self.num_features])     
      Ttrain  = np.zeros([window, n_train, self.num_features])     

      for ff in np.arange(self.num_features):
         for ii in np.arange(n_train):
               start_x                  = stride * ii
               end_x                    = start_x + window
               index                    = range(start_x,end_x)
               Ytrain[0:window, ii, ff] = data[index, ff] 
               Ttrain[0:window, ii, ff] = T[index, ff]

      # Define the parameters for the LSTM model
      T_train      = torch.from_numpy(Ttrain).type(torch.Tensor)
      Y_train      = torch.from_numpy(Ytrain).type(torch.Tensor)
      hidden_size  = window

      # Build the LSTM model and train it
      lstm_model = lstm_encoder_decoder.lstm_seq2seq(T_train.shape[2], hidden_size, num_layers)
      _ = lstm_model.train_model(T_train, Y_train, epochs, window, batch_size)

      # Plot the results of the trained LSTM model         
      self.plot_lstm(n_train, window, stride, Ttrain, Ytrain, lstm_model)

      return lstm_model

   def plot_lstm(self, n_train, window, stride, Ttrain, Ytrain, lstm_model):

      # Plot the results of the trained LSTM model         
      count_dc = 0

      for dc in self.dc_list:
         # Initialize the arrays for the target, input, and output signals
         X = np.zeros([int(n_train*window/self.num_dc)]) 
         Y = np.zeros([int(n_train*window/self.num_dc)])     
         T = np.zeros([int(n_train*window/self.num_dc)])     
         num_samples_per_dc = int(n_train/self.num_dc) 

         for ii in range(num_samples_per_dc):
               start        = ii*stride
               end          = start + window
               train_plt    = Ttrain[:, count_dc*num_samples_per_dc+ii, :]
               Y_train_pred = \
                  lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), \
                     target_len = window)
               X[start:end] = Ytrain[:, count_dc*num_samples_per_dc+ii, 0]
               Y[start:end] = Y_train_pred[:, 0]
               T[start:end] = Ttrain[:, count_dc*num_samples_per_dc+ii, 0]

         # Plot the target and predicted output signals
         plt.rcParams.update({'font.size': 10})
         plt.figure()
         plt.plot(T, X, '-', linewidth = 1.0, markersize = 1.0, label = 'Target')
         # plt.plot(T, Y, '-', linewidth = 1.0, markersize = 1.0, label = 'Reconstructed')
         plt.xlabel('$Time (sec)$')
         plt.ylabel('$a (\mu m/s^2)$')
         plt.legend(frameon=False)
         plt.suptitle('%s data set for dc=%s $\mu m$' % ('Training',dc))
         plt.tight_layout()

         count_dc += 1

         # Free up memory by deleting the arrays
         del X,Y,T
         
      return
   
   def inference(self,nsamples,reduction=False):
      """ 
      Run the MCMC algorithm to estimate the posterior distribution of the model parameters
      Plot the posterior distribution of the model parameters   
      """
      noisy_acc = load_object(self.data_file)  # Load a saved data file
      if reduction:
         model_lstm = self.build_lstm()  # Return the lSTM model
         
      for index, dc in enumerate(self.dc_list):
         start = index*self.model.num_tsteps
         end = start + self.model.num_tsteps
         noisy_data = noisy_acc[start:end]
         print('--- Dc is %s ---' % dc)

         # Perform MCMC sampling without reduction
         MCMCobj = MCMC(
            self.model,
            noisy_data,
            self.qpriors,
            self.qstart,
            nsamples=nsamples)
         qparams = MCMCobj.sample()
         self.plot_dist(qparams, dc)         

         # Perform MCMC sampling with reduction using LSTM model
         if reduction:
            MCMCobj = MCMC(
               self.model,
               noisy_data,
               self.qpriors,
               self.qstart,
               lstm_model=model_lstm,
               nsamples=nsamples)
            qparams = MCMCobj.sample()
            self.plot_dist(qparams, dc)

      return

   def plot_dist(self, qparams, dc):

      # Set up the plot layout with 1 row and 2 columns, 
      # and adjust the width ratios and space between subplots
      n_rows = 1
      n_columns = 2
      gridspec = {'width_ratios': [0.7, 0.15], 'wspace': 0.15}

      # Create the subplots with the specified gridspec
      fig, ax = plt.subplots(n_rows, n_columns, gridspec_kw=gridspec)

      # Set the main plot title with the specified dc value
      fig.suptitle(f'$d_c={dc}\,\mu m$', fontsize=10)

      # Plot the MCMC samples as a blue line in the first subplot
      ax[0].plot(qparams[0, :], 'b-', linewidth=1.0)

      # Get the limits of the y-axis
      ylims = ax[0].get_ylim()

      # Create an array of 1000 evenly spaced points between the y-axis limits
      x = np.linspace(ylims[0], ylims[1], 1000)

      # Calculate the probability density function using Gaussian Kernel Density Estimation
      kde = gaussian_kde(qparams[0, :])

      # Plot the probability density function as a blue line in the second subplot
      ax[1].plot(kde.pdf(x), x, 'b-')

      # Fill the area under the probability density function with a light blue color
      ax[1].fill_betweenx(x, kde.pdf(x), np.zeros(x.shape), alpha=0.3)

      # Set the x-axis limits for the second subplot
      ax[1].set_xlim(0, None)

      # Set labels and axis limits for the first subplot
      ax[0].set_ylabel('$d_c$', fontsize=10)
      ax[0].set_xlim(0, qparams.shape[1])
      ax[0].set_xlabel('Sample number')

      # Set the x-axis label for the second subplot and hide the y-axis
      ax[1].set_xlabel('Prob. density')
      ax[1].get_yaxis().set_visible(False)
      ax[1].get_xaxis().set_visible(True)
      ax[1].get_xaxis().set_ticks([])

      return 


