import numpy as np
import lstm_encoder_decoder
from mcmc import MCMC
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time

def measure_execution_time(func):
    """
    Decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return execution_time
    return wrapper

class rsf:
   '''
   Driver class for RSF model
   '''
   def __init__(
      self, 
      number_slip_values=1, 
      lowest_slip_value=1.0, 
      largest_slip_value=1000.0,
      qstart=10.0,
      qpriors=["Uniform", 0.0, 10000.0], 
      plotfigs=False
   ):
      """
      Initialize the class with specified parameters.

      :param number_slip_values: Number of slip values.
      :param lowest_slip_value: The lowest value of slip.
      :param largest_slip_value: The largest value of slip.
      :param qstart: Starting value for a process (specify the process).
      :param qpriors: Prior distributions for a process (specify the process).
      :param plotfigs: Flag to indicate if figures should be plotted.
      """

      self.num_dc = number_slip_values
      self.dc_list = np.linspace(lowest_slip_value, largest_slip_value, self.num_dc)

      self.num_features = 2
      self.plotfigs = plotfigs

      self.qstart = qstart
      self.qpriors = qpriors

      return
   
   def generate_time_series(self):
      """
      Generates a time series for different values of dc in the dc_list.
      Evaluates the model, generates plots, and appends noise-adjusted acceleration data.

      Returns:
         numpy.ndarray: Array of noise-adjusted acceleration data for all dc values.
      """

      # Calculate the total number of entries
      total_entries = len(self.dc_list) * self.model.num_tsteps
      acc_appended_noise = np.zeros(total_entries)

      for index, dc_value in enumerate(self.dc_list):
         # Evaluate the model for the current value of dc
         self.model.Dc = dc_value
         time_series, acceleration, acc_noise = self.model.evaluate()
         self.plot_time_series(time_series, acceleration)  # Generate plots

         # Calculate the start and end indices for the current segment
         start_index = index * self.model.num_tsteps
         end_index = start_index + self.model.num_tsteps

         # Append the noise-adjusted acceleration data
         acc_appended_noise[start_index:end_index] = acc_noise

      return acc_appended_noise

   def plot_time_series(self, time, acceleration):
      """
      Plots the time series of acceleration.

      :param time: Array of time values.
      :param acceleration: Array of acceleration values.
      """
      if not self.plotfigs:
         return

      self._create_plot(time, acceleration)

   def _create_plot(self, time, acceleration):
      """
      Helper method to create the plot.

      :param time: Array of time values.
      :param acceleration: Array of acceleration values.
      """
      plt.figure()
      title = f'$d_c$={self.model.Dc} $\mu m$ RSF solution'
      plt.title(title)
      plt.plot(time, acceleration, linewidth=1.0, label='True')
      plt.xlim(self.model.t_start - 2.0, self.model.t_final)
      plt.xlabel('Time (sec)')
      plt.ylabel('Acceleration $(\mu m/s^2)$')
      plt.grid(True)
      plt.legend()

   def _create_training_sequences(self, data, T, window, stride):
      """
      Create training sequences for the LSTM model.
      """
      n_train = (data.shape[0] - window) // stride + 1
      Y_train = np.zeros([window, n_train, self.num_features])
      T_train = np.zeros([window, n_train, self.num_features])

      for feature_index in range(self.num_features):
         for i in range(n_train):
               start, end = stride * i, stride * i + window
               Y_train[:, i, feature_index] = data[start:end, feature_index]
               T_train[:, i, feature_index] = T[start:end, feature_index]

      return n_train, torch.from_numpy(Y_train).type(torch.Tensor), torch.from_numpy(T_train).type(torch.Tensor)
         
   def build_lstm(self, epochs=20, num_layers=1, batch_size=1):
      """
      Build and train an LSTM model.
      """
      window  = int(self.model.num_tsteps/20)
      stride  = int(window/5)
      T       = self.t_appended.reshape(-1,self.num_features)
      data    = self.acc_appended.reshape(-1,self.num_features)
      
      # Generate the sequences for training
      n_train, Y_train, T_train = self._create_training_sequences(data, T, window, stride)

      # Define the parameters for the LSTM model      
      hidden_size  = window
      lstm_model = lstm_encoder_decoder.lstm_seq2seq(T_train.shape[2], hidden_size, num_layers)

      # Train the model
      _ = lstm_model.train_model(T_train, Y_train, epochs, window, batch_size)

      # Plot the results of the trained LSTM model
      self.plot_lstm(n_train, window, stride, T_train.numpy(), Y_train.numpy(), lstm_model)

      return lstm_model

   def plot_lstm(self, n_train, window, stride, Ttrain, Ytrain, lstm_model):
      """
      Plot the results of the trained LSTM model.

      Parameters:
      n_train (int): Number of training samples.
      window (int): Window size for LSTM.
      stride (int): Stride for window.
      Ttrain (numpy.ndarray): Training time data.
      Ytrain (numpy.ndarray): Training target data.
      lstm_model (Model): Trained LSTM model.
      """

      plt.rcParams.update({'font.size': 10})
      count_dc = 0

      for dc in self.dc_list:
         num_samples_per_dc = int(n_train / self.num_dc)
         X, Y, T = self.initialize_arrays(n_train, window, num_samples_per_dc)

         for ii in range(num_samples_per_dc):
               start = ii * stride
               end = start + window
               train_plt = Ttrain[:, count_dc * num_samples_per_dc + ii, :]
               Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len=window)
               X[start:end] = Ytrain[:, count_dc * num_samples_per_dc + ii, 0]
               Y[start:end] = Y_train_pred[:, 0]
               T[start:end] = Ttrain[:, count_dc * num_samples_per_dc + ii, 0]

         self.plot_signals(T, X, Y, dc)
         count_dc += 1
         del X, Y, T

   def initialize_arrays(self, n_train, window):
      """
      Initialize arrays for target, input, and output signals.

      Returns:
      Tuple of numpy.ndarrays: Initialized arrays X, Y, T.
      """
      array_size = int(n_train * window / self.num_dc)
      return (np.zeros(array_size), np.zeros(array_size), np.zeros(array_size))

   def plot_signals(self, T, X, Y, dc):
      """
      Plot the target and predicted output signals.

      Parameters:
      T (numpy.ndarray): Time data.
      X (numpy.ndarray): Target signal data.
      Y (numpy.ndarray): Predicted signal data.
      dc (float): Displacement current.
      """
      plt.figure()
      plt.plot(T, X, '-', linewidth=1.0, markersize=1.0, label='Target')
      plt.plot(T, Y, '-', linewidth=1.0, markersize=1.0, label='Predicted')
      plt.xlabel('Time (sec)')
      plt.ylabel('a (μm/s²)')
      plt.legend(frameon=False)
      plt.suptitle(f'Training data set for dc={dc} μm')
      plt.tight_layout()
      plt.show()
           
   @measure_execution_time
   def inference(self, data, nsamples, reduction=False):
      """
      Run the MCMC algorithm to estimate the posterior distribution of the model parameters
      Plot the posterior distribution of the model parameters   
      """
      data = self.prepare_data(data)

      # Return the LSTM model
      model_lstm = self.build_lstm() if reduction else None 

      for dc in self.dc_list:
         self.perform_sampling_and_plotting(data, dc, nsamples, reduction, model_lstm)

      return

   def prepare_data(self, data):
      if self.format == 'json':
         from json_save_load import save_object, load_object
         self.lstm_file = 'model_lstm.json'
         self.data_file = 'data.json'
         save_object(data, self.data_file)
         data = load_object(self.data_file)

      elif self.format == 'mysql':
         from mysql_save_load import save_object, load_object
         # MySQL connection details
         mysql_host = 'localhost'
         mysql_user = 'my_user'
         mysql_password = 'my_password'
         mysql_database = 'my_database'
         save_object(data, mysql_host, mysql_user, mysql_password, mysql_database)
         data = load_object(mysql_host, mysql_user, mysql_password, mysql_database)

      return data

   def perform_sampling_and_plotting(self, data, dc, nsamples, reduction, model_lstm):
      # Find the index of 'dc' in the NumPy array 'self.dc_list'
      index = np.where(self.dc_list == dc)[0][0] if dc in self.dc_list else -1
      if index == -1:
         print(f"Error: dc value {dc} not found in dc_list.")
         return

      start = index * self.model.num_tsteps
      end = start + self.model.num_tsteps
      noisy_data = data[start:end]
      print('--- Dc is %s ---' % dc)

      # Perform MCMC sampling without reduction
      MCMCobj = MCMC(self.model, noisy_data, self.qpriors, self.qstart, nsamples=nsamples)
      qparams = MCMCobj.sample()
      self.plot_dist(qparams, dc)

      # Perform MCMC sampling with reduction using LSTM model
      if reduction:
         MCMCobj = MCMC(self.model, noisy_data, self.qpriors, self.qstart, lstm_model=model_lstm, nsamples=nsamples)
         qparams = MCMCobj.sample()
         self.plot_dist(qparams, dc)
        
   def plot_dist(self, qparams, dc):
      """
      Plot the distribution of MCMC samples and their probability density.

      Args:
         qparams: MCMC sample parameters.
         dc: Critical slip distance value.
      """
      # Constants
      KDE_POINTS = 1000
      PLOT_WIDTH_RATIO = [0.7, 0.15]
      PLOT_SPACING = 0.15
      PLOT_LINE_WIDTH = 1.0
      PLOT_ALPHA = 0.3

      # Set up the plot layout
      fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': PLOT_WIDTH_RATIO, 'wspace': PLOT_SPACING})
      fig.suptitle(f'$d_c={dc}\,\mu m$ with {self.format} formatting', fontsize=10)

      # Plot MCMC samples
      axes[0].plot(qparams[0, :], 'b-', linewidth=PLOT_LINE_WIDTH)
      axes[0].set_ylabel('$d_c$', fontsize=10)
      axes[0].set_xlabel('Sample number')
      axes[0].set_xlim(0, qparams.shape[1])

      # Calculate and plot KDE
      kde_x_values = np.linspace(*axes[0].get_ylim(), KDE_POINTS)
      kde = gaussian_kde(qparams[0, :])
      axes[1].plot(kde.pdf(kde_x_values), kde_x_values, 'b-', linewidth=PLOT_LINE_WIDTH)
      axes[1].fill_betweenx(kde_x_values, kde.pdf(kde_x_values), np.zeros(kde_x_values.shape), alpha=PLOT_ALPHA)
      axes[1].set_xlim(0, None)
      axes[1].set_xlabel('Prob. density')
      axes[1].get_yaxis().set_visible(False)
      axes[1].get_xaxis().set_visible(True)
      axes[1].get_xaxis().set_ticks([])

      return
