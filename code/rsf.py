from RateStateModel import RateStateModel
import numpy as np
from save_load import save_object, load_object
import lstm_encoder_decoder
from MCMC import MCMC
import torch
import matplotlib.pyplot as plt


class rsf:
   '''
   Driver class for RSF model
   '''
   def __init__(
      self, 
      number_slip_values=1, 
      lowest_slip_value=1.0, 
      largest_slip_value=1000.0):
      
      # Define the range of values for the critical slip distance
      self.num_dc       = number_slip_values
      start_dc          = lowest_slip_value
      end_dc            = largest_slip_value
      self.dc_list      = np.linspace(start_dc,end_dc,self.num_dc)

      # Define file names for saving and loading data and LSTM model
      self.lstm_file    = 'model_lstm.pickle'
      self.data_file    = 'data.pickle'
      self.num_features = 2

      return
   
   def time_series(self):

      # rate state model
      self.model                         = RateStateModel()
      # Create arrays to store the time and acceleration data for all values of dc
      t_appended                         = np.zeros((self.num_dc*self.model.num_tsteps,self.num_features))
      acc_appended                       = np.zeros((self.num_dc*self.model.num_tsteps,self.num_features))
      acc_appended_noise                 = np.zeros((self.num_dc*self.model.num_tsteps,self.num_features))  
      count_dc                           = 0
      for dc in self.dc_list:
         # Evaluate the model for the current value of dc
         self.model.Dc                   = dc
         t, acc, acc_noise               = self.model.evaluate() # noisy data
         # Generate plots if desired
         self.generateplots(t[:,0], acc[:,0], acc_noise[:,0])      
         # Append the time and acceleration data to the corresponding arrays
         start                           = count_dc*self.model.num_tsteps
         end                             = start + self.model.num_tsteps
         t_appended[start:end,0]         = t[:,0]
         t_appended[start:end,1]         = dc
         acc_appended[start:end,0]       = acc[:,0]
         acc_appended[start:end,1]       = acc[:,1]
         acc_appended_noise[start:end,0] = acc_noise[:,0]
         acc_appended_noise[start:end,1] = acc_noise[:,1]
         count_dc += 1

      # Store the time and acceleration data as attributes of the class
      self.t_appended                    = t_appended
      self.acc_appended                  = acc_appended

      # Save the data using pickle
      save_object(acc_appended_noise,self.data_file)

      return
         
   def build_lstm(self):
      """ 
      Prepare the training data for the LSTM model
      """
      t_                      = self.t_appended.reshape(-1,self.num_features)
      var_                    = self.acc_appended.reshape(-1,self.num_features)
      n_train, Ttrain, Ytrain = self.windowed_dataset(t_, var_, 
                                                      self.model.window, 
                                                      self.model.stride, 
                                                      self.num_features)
      T_train                 = torch.from_numpy(Ttrain).type(torch.Tensor)
      Y_train                 = torch.from_numpy(Ytrain).type(torch.Tensor)

      # Define the parameters for the LSTM model
      hidden_size             = self.model.window
      batch_size              = 1
      n_epochs                = self.model.epochs
      num_layers              = 1 
      input_tensor            = T_train

      # Build the LSTM model and train it
      model_lstm              = lstm_encoder_decoder.lstm_seq2seq(input_tensor.shape[2], 
                                                                  hidden_size, 
                                                                  num_layers, False)
      loss                    = model_lstm.train_model(input_tensor, 
                                                      Y_train, 
                                                      n_epochs, 
                                                      self.model.window, 
                                                      batch_size)

      # Save the trained LSTM model to a file
      save_object(model_lstm,self.lstm_file) 

      # Plot the results of the trained LSTM model
      self.plot_results(
         model_lstm, 
         Ttrain, 
         Ttrain, 
         Ytrain, 
         self.model.stride, 
         self.model.window, 
         'Training', 
         'Reconstruction', 
         n_train, 
         self.num_dc, 
         self.dc_list, 
         self.model.num_tsteps)

      return

   def windowed_dataset(self, t, y, window, stride, num_features = 1):
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
      L                            = y.shape[0]
      num_samples                  = (L - window) // stride + 1
      Y                            = np.zeros([window, num_samples, num_features])     
      T                            = np.zeros([window, num_samples, num_features])     
      for ff in np.arange(num_features):
         for ii in np.arange(num_samples):
               start_x             = stride * ii
               end_x               = start_x + window
               index               = range(start_x,end_x)
               Y[0:window, ii, ff] = y[index, ff] 
               T[0:window, ii, ff] = t[index, ff]

      return num_samples, T, Y
            
   def plot_results(
      self,lstm_model, T_, X_, Y_, 
      stride, window, dataset_type, 
      objective, num_samples, num_p, p_, num_tsteps):
         
      # Initialize the reconstructed signal array
      Y_return              = np.zeros([int(num_samples*window)])     
      count_dc              = 0
      for dc in p_:
         # Initialize the arrays for the target, input, and output signals
         X                  = np.zeros([int(num_samples*window/num_p)]) 
         Y                  = np.zeros([int(num_samples*window/num_p)])     
         T                  = np.zeros([int(num_samples*window/num_p)])     
         num_samples_per_dc = int(num_samples/num_p) 
         # Iterate through the samples and generate the predicted output signal for the current DC value
         for ii in range(num_samples_per_dc):
               start        = ii*stride
               end          = start + window
               train_plt    = X_[:, count_dc*num_samples_per_dc+ii, :]
               Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len = window)
               X[start:end] = Y_[:, count_dc*num_samples_per_dc+ii, 0]
               Y[start:end] = Y_train_pred[:, 0]
               T[start:end] = T_[:, count_dc*num_samples_per_dc+ii, 0]
               Y_return[count_dc*num_tsteps+start:count_dc*num_tsteps+end] = Y_train_pred[:, 0]

         # Plot the target and predicted output signals
         plt.rcParams.update({'font.size': 10})
         plt.figure()
         plt.plot(T, X, '-', color = (0.2, 0.42, 0.72), linewidth = 1.0, markersize = 1.0, label = 'Target')
         plt.plot(T, Y, '-', color = (0.76, 0.01, 0.01), linewidth = 1.0, markersize = 1.0, label = '%s' % objective)
         plt.xlabel('$Time (sec)$')
         plt.ylabel('$a (\mu m/s^2)$')
         plt.legend(frameon=False)
         plt.suptitle('%s data set for dc=%s $\mu m$' % (dataset_type,dc), x = 0.445, y = 1.)
         plt.tight_layout()

         count_dc += 1

         # Free up memory by deleting the arrays
         del X,Y,T

      return
         
   def inference_full(self,nsamples):

      # Load data
      noisy_acc  = load_object(self.data_file)
      # Loop over each duty cycle and perform Bayesian inference
      for j in range(self.num_dc):
         # Extract the data for this duty cycle and reshape it to a 1D array
         data    = noisy_acc[j*self.model.num_tsteps:j*self.model.num_tsteps+self.model.num_tsteps,0]
         data    = data.reshape(1, self.model.num_tsteps)
         # Extract the duty cycle and print a message
         dc      = self.dc_list[j]
         print(f'--- d_c is {dc}')
         # Set up the prior and initial guess for the Dc parameter
         qstart  = 0.1
         qpriors = ["Uniform",0.1, 1000]
         # Run the MCMC algorithm to estimate the posterior distribution of the model parameters
         MCMCobj = MCMC(self.model,data,qpriors,qstart,adapt_interval=10,nsamples=nsamples)
         qparams = MCMCobj.sample()       
         # Plot the posterior distribution of the model parameters
         MCMCobj.plot_dist(qparams,dc)

      return
   
   def rsf_inference(self,nsamples):

      model_lstm         = load_object(self.lstm_file)  # Load a saved LSTM model
      acc_appended_noise = load_object(self.data_file)  # Load a saved data file
      num_p              = self.num_dc  # Get the number of parameters
      p_                 = self.dc_list  # Get the parameter values
      num_tsteps         = self.model.num_tsteps  # Get the number of time steps
      for ii in range(0,num_p):
            # Get the accelerometer data for the current parameter index
            acc          = acc_appended_noise[ii*num_tsteps:ii*num_tsteps+num_tsteps,0]
            acc          = acc.reshape(1, num_tsteps)
            dc           = p_[ii]  # Get the current parameter value
            print('--- dc is %s ---' % dc)
            # Set up the prior and initial guess for the Dc parameter
            qstart       = 0.1
            qpriors      = ["Uniform",0.1, 1000]
            # Run the MCMC algorithm to estimate the posterior distribution of the model parameters
            MCMCobj      = MCMC(self.model,acc,qpriors,qstart,nsamples=nsamples)
            qparams      = MCMCobj.sample()
            # Plot the posterior distribution of the model parameters
            MCMCobj.plot_dist(qparams,dc)
            # Create an MCMC object for the reduced-order model and run the algorithm
            MCMCobj      = MCMC(self.model,acc,qpriors,qstart,lstm_model=model_lstm,nsamples=nsamples)
            qparams      = MCMCobj.sample()
            # Plot the posterior distribution of the parameter values for the reduced-order model
            MCMCobj.plot_dist(qparams,dc)

      return
   
   def generateplots(self,t,acc,acc_noise):

      if self.plotfigs:
         # Plots
         plt.figure()
         plt.title('$d_c$=' + str(self.model.Dc) + ' $\mu m$' + ' RSF solution')
         plt.plot(t, acc, '--', linewidth=1.0,label='True')
         plt.plot(t, acc_noise, linewidth=1.0,label='Noisy')
         plt.xlim(self.model.t_start - 2.0, self.model.t_final)
         plt.xlabel('Time (sec)')
         plt.ylabel('Acceleration $(\mu m/s^2)$')
         plt.grid('on')
         plt.legend()        
      
      return        
   
   
