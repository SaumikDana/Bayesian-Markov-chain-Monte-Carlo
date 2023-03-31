from RateStateModel import RateStateModel
import numpy as np
import math
from save_load import save_object, load_object
import generate_dataset
import lstm_encoder_decoder
from MCMC import MCMC
import sys
import torch
import matplotlib.pyplot as plt


class rsf:
   '''
   Driver class for RSF model
   '''
   def __init__(self, number_slip_values=1, lowest_slip_value=1.0, largest_slip_value=1000.0):
      
      # Define the range of values for the critical slip distance
      self.num_p = number_slip_values
      start_dc = lowest_slip_value
      end_dc = largest_slip_value
      self.dc_list = np.logspace(math.log10(start_dc),math.log10(end_dc),self.num_p)

      # Define file names for saving and loading data and LSTM model
      self.lstm_file = 'model_lstm.pickle'
      self.data_file = 'data.pickle'
      self.num_features = 2

      return
   
   def time_series(self):

      # rate state model
      self.model = RateStateModel()

      # Create arrays to store the time and acceleration data for all values of dc
      t_appended = np.zeros((self.num_p*self.model.num_tsteps,self.num_features))
      acc_appended = np.zeros((self.num_p*self.model.num_tsteps,self.num_features))
      acc_appended_noise = np.zeros((self.num_p*self.model.num_tsteps,self.num_features))
      
      count_dc = 0
      for dc in self.dc_list:
         # Evaluate the model for the current value of dc
         self.model.set_dc(dc)
         t, acc, acc_noise = self.model.evaluate() # noisy data
         # Generate plots if desired
         self.generateplots(t[:,0], acc[:,0], acc_noise[:,0])      

         # Append the time and acceleration data to the corresponding arrays
         start = count_dc*self.model.num_tsteps
         end = start + self.model.num_tsteps
         t_appended[start:end,0] = t[:,0]
         t_appended[start:end,1] = dc
         acc_appended[start:end,0] = acc[:,0]
         acc_appended[start:end,1] = acc[:,1]
         acc_appended_noise[start:end,0] = acc_noise[:,0]
         acc_appended_noise[start:end,1] = acc_noise[:,1]
         count_dc += 1

      # Store the time and acceleration data as attributes of the class
      self.t_appended = t_appended
      self.acc_appended = acc_appended

      # Save the data using pickle
      save_object(acc_appended_noise,self.data_file)

      return
         
   def build_lstm(self):
      # Prepare the training data for the LSTM model
      t_ = self.t_appended.reshape(-1,self.num_features)
      var_ = self.acc_appended.reshape(-1,self.num_features)
      num_samples_train, Ttrain, Ytrain = generate_dataset.windowed_dataset(
         t_, var_, self.window, self.stride, self.num_features) 
      T_train, Y_train = generate_dataset.numpy_to_torch(Ttrain, Ytrain)

      # Define the parameters for the LSTM model
      hidden_size = self.window
      batch_size = 1
      n_epochs = int(sys.argv[1])
      num_layers = 1 
      input_tensor = T_train

      # Build the LSTM model and train it
      model_lstm = lstm_encoder_decoder.lstm_seq2seq(
         input_tensor.shape[2], hidden_size, num_layers, False)
      loss = model_lstm.train_model(
         input_tensor, Y_train, n_epochs, self.window, batch_size)

      # Save the trained LSTM model to a file
      save_object(model_lstm,self.lstm_file) 

      # Plot the results of the trained LSTM model
      self.plot_results(
         model_lstm, Ttrain, Ttrain, Ytrain, 
         self.stride, self.window, 
         'Training', 'Reconstruction', 
         num_samples_train, self.num_p, self.dc_list, self.num_tsteps)

      return
      
   def plot_results(
      self,lstm_model, T_, X_, Y_, 
      stride, window, dataset_type, 
      objective, num_samples, num_p, p_, num_tsteps):
         
      # Initialize the reconstructed signal array
      Y_return = np.zeros([int(num_samples*window)])     

      count_dc = 0
      for dc in p_:
         # Initialize the arrays for the target, input, and output signals
         X = np.zeros([int(num_samples*window/num_p)]) 
         Y = np.zeros([int(num_samples*window/num_p)])     
         T = np.zeros([int(num_samples*window/num_p)])     

         num_samples_per_dc = int(num_samples/num_p) 

         # Iterate through the samples and generate the predicted output signal for the current DC value
         for ii in range(num_samples_per_dc):
               start = ii*stride
               end = start + window

               train_plt = X_[:, count_dc*num_samples_per_dc+ii, :]
               Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len = window)

               X[start:end] = Y_[:, count_dc*num_samples_per_dc+ii, 0]
               Y[start:end] = Y_train_pred[:, 0]
               T[start:end] = T_[:, count_dc*num_samples_per_dc+ii, 0]
               Y_return[count_dc*num_tsteps+start:count_dc*num_tsteps+end] = Y_train_pred[:, 0]

         # Plot the target and predicted output signals
         plt.rcParams.update({'font.size': 16})
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

         # Free up memory by deleting the arrays
         del X,Y,T

      return
         
   def rsf_inference_no_rom(self):

      # Load data
      acc_appended_noise = load_object(self.data_file)

      for j in range(self.num_p):
         start = j*self.model.num_tsteps
         end = start + self.model.num_tsteps
         acc = acc_appended_noise[start:end,0]
         acc = acc.reshape(1, self.model.num_tsteps)
      
         dc = self.dc_list[j]
         print('--- dc is %s ---' % dc)

         qstart={"Dc":100} # set initial guess for Dc parameter
         qpriors={"Dc":["Uniform",0.1, 1000]} # set priors for Dc parameter
                  
         # run Bayesian/MCMC algorithm
         MCMCobj = MCMC(self.model,acc,qpriors,qstart,adapt_interval=10)
         qparams = MCMCobj.sample() 
         
         # plot posterior distribution
         MCMCobj.plot_dist(qparams,dc)

      return
   
   def rsf_inference(self):
      # load reduced order model!!!
      model_lstm = load_object(self.lstm_file)  # Load a saved LSTM model

      # load data!!!
      acc_appended_noise = load_object(self.data_file)  # Load a saved data file

      num_p = self.num_p  # Get the number of parameters
      p_ = self.dc_list  # Get the parameter values
      num_tsteps = self.num_tsteps  # Get the number of time steps
      model = self.model  # Get the physics-based model

      for ii in range(0,num_p):
         # Check if the current parameter index is in a list of indices
         if ii == 2 or ii == 11 or ii == 13 or ii == 15 or ii == 16 or ii == 17:
               # Get the accelerometer data for the current parameter index
               acc = acc_appended_noise[ii*num_tsteps:ii*num_tsteps+num_tsteps,0]
               acc = acc.reshape(1, num_tsteps)

               dc = p_[ii]  # Get the current parameter value
               print('--- dc is %s ---' % dc)

               qstart={"Dc":100}  # Define the initial guess for the parameter values
               qpriors={"Dc":["Uniform",0.1, 1000]}  # Define the prior distribution for the parameter values

               nsamples = int(sys.argv[2])  # Get the number of MCMC samples
               nburn = nsamples/2  # Define the number of burn-in samples

               problem_type = 'full'
               # Create an MCMC object for the full physics-based model and run the algorithm
               MCMCobj1=MCMC(model,qpriors=qpriors,nsamples=nsamples,nburn=nburn,data=acc,problem_type=problem_type,lstm_model=model_lstm,qstart=qstart,adapt_interval=10,verbose=True)
               qparams1=MCMCobj1.sample()

               # Plot the posterior distribution of the parameter values for the full physics-based model
               MCMCobj1.plot_dist(qparams1,'full',dc)

               problem_type = 'rom'
               # Create an MCMC object for the reduced-order model and run the algorithm
               MCMCobj2=MCMC(model,qpriors=qpriors,nsamples=nsamples,nburn=nburn,data=acc,problem_type=problem_type,lstm_model=model_lstm,qstart=qstart,adapt_interval=10,verbose=True)
               qparams2=MCMCobj2.sample()

               # Plot the posterior distribution of the parameter values for the reduced-order model
               MCMCobj2.plot_dist(qparams2,'reduced order',dc)

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
   
   
