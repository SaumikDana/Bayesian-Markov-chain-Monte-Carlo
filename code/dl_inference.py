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

from pathlib import Path


def build_lstm(t_appended, var_appended, problem_):
 
    # build rom!!!
    t_ = t_appended.reshape(-1,problem_.num_features); var_ = var_appended.reshape(-1,problem_.num_features)
    num_samples_train, Ttrain, Ytrain = generate_dataset.windowed_dataset(t_, var_, problem_.window, problem_.stride, problem_.num_features) 
    T_train, Y_train = generate_dataset.numpy_to_torch(Ttrain, Ytrain)
    hidden_size = problem_.window; batch_size = 1; n_epochs = int(sys.argv[1]); num_layers = 1 
    input_tensor = T_train
    model_lstm = lstm_encoder_decoder.lstm_seq2seq(input_size = input_tensor.shape[2], hidden_size = hidden_size, num_layers = num_layers, bidirectional = False)
    loss = model_lstm.train_model(input_tensor, Y_train, n_epochs = n_epochs, target_len = window, batch_size = batch_size, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)

    # save objects!!!
    my_file = Path(os.getcwd()+'/model_lstm.pickle')
    if not my_file.is_file():
       save_object(model_lstm,"model_lstm.pickle") 

    # plot!!!
    problem_.plot_results(model_lstm, Ttrain, Ttrain, Ytrain, problem_.stride, problem_.window, 'Training', 'Reconstruction', num_samples_train, problem_.num_p, problem_.p_, problem_.num_tsteps)

    return my_file

    
def main(problem,rom,bayesian):

    # rsf problem!!!
    if problem == 'rsf':
       problem_ = rsf()
       t_appended, acc_appended, file2 = problem_.time_series() 

       if rom:
       # LSTM encoder-decoder!!!
         file1 = build_lstm(t_appended,acc_appended,problem_)

       # bayesian!!!
       if bayesian:
          if rom:
             rsf_inference(file1,file2,problem_.num_p,problem_.num_tsteps,problem_.p_,problem_.model)      
          else:
             rsf_inference_no_rom(file1,file2,problem_.num_p,problem_.num_tsteps,problem_.p_,problem_.model)      

    # pylith-gprs problem!!!
    elif problem == 'pylith_gprs':
       problem_ = pylith_gprs()
       t_appended, u_appended, file2 = problem_.time_series() 

       if rom:   
       # LSTM encoder-decoder!!!
          file1 = build_lstm(t_appended,u_appended,problem_)

       # bayesian!!!
       if bayesian:
          pylith_gprs_inference(file1,file2,problem_.num_p,problem_.num_tsteps,problem_.p_)      

    # Close it out!!!
    plt.show()
    plt.close('all')


# Driver code!!!
if __name__ == '__main__':

    problem = 'rsf'
#    problem = 'pylith_gprs'
    rom = False
#    rom = True
#    bayesian = False
    bayesian = True
    main(problem,rom,bayesian)




