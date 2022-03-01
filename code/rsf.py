from RateStateModel import RateStateModel
import numpy as np
import math
from save_load import save_object, load_object
import generate_dataset
import lstm_encoder_decoder
from MCMC import MCMC
import sys
import os
import matplotlib.pyplot as plt


class rsf:
   '''
   Driver class for RSF model
   '''
   def __init__(self,args):

       self.num_p = 1
       start_dc = 1.0
       end_dc = 1000.0
       self.p_ = np.logspace(math.log10(start_dc),math.log10(end_dc),self.num_p)

       t_start = 0.0
       t_end = 50.0
       self.num_tsteps = 500

       self.window = 25 # make sure num_tsteps is exact multiple of window!!!
       self.stride = 25
       self.num_features = 2

       self.model = RateStateModel(t_start, t_end, num_tsteps = self.num_tsteps, window = self.window) # forward model
       self.args = args

       self.lstm_file = 'model_lstm.pickle'
       self.data_file = 'data.pickle'
 

   def solve(self,args):

       self.time_series() 

       if self.args.reduction:
       # LSTM encoder-decoder!!!
         self.build_lstm()

       if self.args.bayesian:
       # bayesian!!!
          if self.args.reduction:
             self.rsf_inference()      
          else:
             self.rsf_inference_no_rom()      


   def time_series(self):

       num_p = self.num_p
       p_ = self.p_
       num_tsteps = self.num_tsteps
       num_features = self.num_features
       model = self.model
       num_features = self.num_features

       t_appended =  np.zeros((num_p*num_tsteps,num_features))
       acc_appended =  np.zeros((num_p*num_tsteps,num_features))
       acc_appended_noise = np.zeros((num_p*num_tsteps,num_features))
       
       count_dc = 0
       for dc in p_:
          model.set_dc(dc)
          t, acc, acc_noise = model.evaluate(model.consts) # noisy data
          t_ = t.reshape(-1,num_features); acc_ = acc.reshape(-1,num_features)
          start_ = count_dc*num_tsteps; end_ = start_ + num_tsteps
          t_appended[start_:end_,0] = t[:,0]; t_appended[start_:end_,1] = dc
          acc_appended[start_:end_,0] = acc[:,0]; acc_appended[start_:end_,1] = acc[:,1]
          acc_appended_noise[start_:end_,0] = acc_noise[:,0]; acc_appended_noise[start_:end_,1] = acc_noise[:,1]
          count_dc += 1

  
       self.t_appended = t_appended
       self.acc_appended = acc_appended
 
       # pickle data!!!
       save_object(acc_appended_noise,self.data_file)

 
   def build_lstm(self):
    
       # build rom!!!
       t_ = self.t_appended.reshape(-1,self.num_features); var_ = self.acc_appended.reshape(-1,self.num_features)
       num_samples_train, Ttrain, Ytrain = generate_dataset.windowed_dataset(t_, var_, self.window, self.stride, self.num_features) 
       T_train, Y_train = generate_dataset.numpy_to_torch(Ttrain, Ytrain)
       hidden_size = self.window; batch_size = 1; n_epochs = int(sys.argv[1]); num_layers = 1 
       input_tensor = T_train
       model_lstm = lstm_encoder_decoder.lstm_seq2seq(input_tensor.shape[2], hidden_size, num_layers, False)
       loss = model_lstm.train_model(input_tensor, Y_train, n_epochs, self.window, batch_size)
   
       # pickle reduced order model!!!
       save_object(model_lstm,self.lstm_file) 
   
       # plot!!!
       self.plot_results(model_lstm, Ttrain, Ttrain, Ytrain, self.stride, self.window, 'Training', 'Reconstruction', num_samples_train, self.num_p, self.p_, self.num_tsteps)

   
   def plot_results(self,lstm_model, T_, X_, Y_, stride, window, dataset_type, objective, num_samples, num_p, p_, num_tsteps):
     '''
     plot examples of the lstm encoder-decoder evaluated on the training/test data
     
     '''
     Y_return = np.zeros([int(num_samples*window)])     
     count_dc = 0
     for dc in dc_:
   
         X = np.zeros([int(num_samples*window/num_p)]) 
         Y = np.zeros([int(num_samples*window/num_p)])     
         T = np.zeros([int(num_samples*window/num_p)])     
   
         num_samples_per_dc = int(num_samples/num_p) 
   
         for ii in range(num_samples_per_dc):
   
             start = ii*stride
             end = start + window
   
             train_plt = X_[:, count_dc*num_samples_per_dc+ii, :]
             Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len = window)
   
             X[start:end] = Y_[:, count_dc*num_samples_per_dc+ii, 0]
             Y[start:end] = Y_train_pred[:, 0]
             T[start:end] = T_[:, count_dc*num_samples_per_dc+ii, 0]
             Y_return[count_dc*num_tsteps+start:count_dc*num_tsteps+end] = Y_train_pred[:, 0]
   
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
   
         del X,Y,T
   
   
   def rsf_inference_no_rom(self):

       # load reduced order model!!!
       model_lstm = load_object(self.lstm_file)

       # load data!!!
       acc_appended_noise = load_object(self.data_file)

       num_p = self.num_p
       p_ = self.p_
       num_tsteps = self.num_tsteps
       model = self.model
  
       for ii in range(0,num_p):
  
          # noisy data!!!
          acc = acc_appended_noise[ii*num_tsteps:ii*num_tsteps+num_tsteps,0]
          acc = acc.reshape(1, num_tsteps)
       
          dc = p_[ii]
          print('--- dc is %s ---' % dc)
  
          qstart={"Dc":100} # initial guess
          qpriors={"Dc":["Uniform",0.1, 1000]}
  
          nsamples = int(sys.argv[2])
          nburn = nsamples/2
          
          problem_type = 'full'
          MCMCobj1=MCMC(model,qpriors=qpriors,nsamples=nsamples,nburn=nburn,data=acc,problem_type=problem_type,lstm_model=model_lstm,qstart=qstart,adapt_interval=10,verbose=True)
          qparams1=MCMCobj1.sample() # run the Bayesian/MCMC algorithm
          MCMCobj1.plot_dist(qparams1,'full',dc)
  

   def rsf_inference(self):

       # load reduced order model!!!
       model_lstm = load_object(self.lstm_file)

       # load data!!!
       acc_appended_noise = load_object(self.data_file)

       num_p = self.num_p
       p_ = self.p_
       num_tsteps = self.num_tsteps
       model = self.model
   
       for ii in range(0,num_p):
   
           if ii == 2 or ii == 11 or ii == 13 or ii == 15 or ii == 16 or ii == 17:
              # noisy data!!!
              acc = acc_appended_noise[ii*num_tsteps:ii*num_tsteps+num_tsteps,0]
              acc = acc.reshape(1, num_tsteps)
        
              dc = p_[ii]
              print('--- dc is %s ---' % dc)
   
              qstart={"Dc":100} # initial guess
              qpriors={"Dc":["Uniform",0.1, 1000]}
   
              nsamples = int(sys.argv[2])
              nburn = nsamples/2
              
              problem_type = 'full'
              MCMCobj1=MCMC(model,qpriors=qpriors,nsamples=nsamples,nburn=nburn,data=acc,problem_type=problem_type,lstm_model=model_lstm,qstart=qstart,adapt_interval=10,verbose=True)
              qparams1=MCMCobj1.sample() # run the Bayesian/MCMC algorithm
              MCMCobj1.plot_dist(qparams1,'full',dc)
   
              problem_type = 'rom'
              MCMCobj2=MCMC(model,qpriors=qpriors,nsamples=nsamples,nburn=nburn,data=acc,problem_type=problem_type,lstm_model=model_lstm,qstart=qstart,adapt_interval=10,verbose=True)
              qparams2=MCMCobj2.sample() # run the Bayesian/MCMC algorithm
              MCMCobj2.plot_dist(qparams2,'reduced order',dc)
   
   
   
