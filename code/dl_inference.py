from rsf import rsf
from pylith_gprs import pylith_gprs
from inference import rsf_inference,rsf_inference_no_rom,pylith_gprs_inference
import numpy as np
from MCMC import MCMC
import sys
import os
import matplotlib.pyplot as plt
from save_load import save_object, load_object

#import generate_dataset
#import lstm_encoder_decoder
#import plotting_time_series 

from pathlib import Path

def main(problem,rom,bayesian):

    # rsf problem!!!
    if problem == 'rsf':
       problem_ = rsf()
       t_appended, acc_appended, acc_appended_noise = problem_.time_series() 

       # save objects!!!
       my_file = Path(os.getcwd()+'/acc_appended_noise.pickle')
       if not my_file.is_file():
          save_object(acc_appended_noise,"acc_appended_noise.pickle")
    
       # LSTM encoder-decoder!!!
       if rom:
          num_features = problem_.num_features
          window = problem_.window
          stride = problem_.stride
          num_dc = problem_.num_dc
          num_tsteps = problem_.num_tsteps
          dc_ = problem_.dc_
          t_ = t_appended.reshape(-1,num_features); acc_ = acc_appended.reshape(-1,num_features)
          num_samples_train, Ttrain, Ytrain = generate_dataset.windowed_dataset(t_, acc_, window, stride, num_features) 
          T_train, Y_train = generate_dataset.numpy_to_torch(Ttrain, Ytrain)
          hidden_size = window; batch_size = 1; n_epochs = int(sys.argv[1]); num_layers = 1 
          input_tensor = T_train
          model_lstm = lstm_encoder_decoder.lstm_seq2seq(input_size = input_tensor.shape[2], hidden_size = hidden_size, num_layers = num_layers, bidirectional = False)
          loss = model_lstm.train_model(input_tensor, Y_train, n_epochs = n_epochs, target_len = window, batch_size = batch_size, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)

          # save objects!!!
          my_file = Path(os.getcwd()+'/model_lstm.pickle')
          if not my_file.is_file():
             save_object(model_lstm,"model_lstm.pickle") 

          # plot!!!
          plotting_time_series.plot_train_test_results(model_lstm, Ttrain, Ttrain, Ytrain, stride, window, 'Training', 'Reconstruction', num_samples_train, num_dc, dc_, num_tsteps)

       # bayesian!!!
       if bayesian:
          file1 = 'model_lstm.pickle'
          file2 = 'acc_appended_noise.pickle'
          if rom:
             rsf_inference(file1,file2,problem_.num_dc,problem_.num_tsteps,problem_.dc_,problem_.model)      
          else:
             rsf_inference_no_rom(file1,file2,problem_.num_dc,problem_.num_tsteps,problem_.dc_,problem_.model)      

    # pylith-gprs problem!!!
    elif problem == 'pylith_gprs':
       problem_ = pylith_gprs()
       t_appended, ux_appended, uy_appended = problem_.time_series() 
   
       # save objects!!! 
       my_file = Path(os.getcwd()+'/ux_appended.pickle')
       if not my_file.is_file():
          save_object(ux_appended,"ux_appended.pickle")

       # LSTM encoder-decoder!!!
       if rom:   
          num_features = problem_.num_features
          window = problem_.window
          stride = problem_.stride
          num_q = problem_.num_q
          num_tsteps = problem_.num_tsteps
          q_ = problem_.q_
          t_ = t_appended.reshape(-1,num_features); ux_ = ux_appended.reshape(-1,num_features)
          num_samples_train, Ttrain, Ytrain = generate_dataset.windowed_dataset(t_, ux_, window, stride, num_features) 
          T_train, Y_train = generate_dataset.numpy_to_torch(Ttrain, Ytrain)
          hidden_size = window; batch_size = 1; n_epochs = int(sys.argv[1]); num_layers = 1 
          input_tensor = T_train
          model_lstm = lstm_encoder_decoder.lstm_seq2seq(input_size = input_tensor.shape[2], hidden_size = hidden_size, num_layers = num_layers, bidirectional = False)
          loss = model_lstm.train_model(input_tensor, Y_train, n_epochs = n_epochs, target_len = window, batch_size = batch_size, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)

          # save objects!!!
          my_file = Path(os.getcwd()+'/model_lstm.pickle')
          if not my_file.is_file():
             save_object(model_lstm,"model_lstm.pickle") 

          # plot!!!
          plotting_time_series.plot_results(model_lstm, Ttrain, Ttrain, Ytrain, stride, window, 'Training', 'Reconstruction', num_samples_train, num_q, q_, num_tsteps)

       # bayesian!!!
       if bayesian:
          file1 = 'model_lstm.pickle'
          file2 = 'ux_appended.pickle'
          pylith_gprs_inference(file1,file2,problem_.num_q,problem_.num_tsteps,problem_.q_)      

    # Close it out!!!
    plt.show()
    plt.close('all')

if __name__ == '__main__':

    problem = 'rsf'
#    problem = 'pylith_gprs'
    rom = False
#    rom = True
#    bayesian = False
    bayesian = True
    main(problem,rom,bayesian)




