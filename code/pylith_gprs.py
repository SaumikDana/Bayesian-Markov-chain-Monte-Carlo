from parse_vtk import parse_vtk
import numpy as np
import os
from pathlib import Path
from save_load import save_object, load_object
import generate_dataset
import lstm_encoder_decoder
from MCMC import MCMC
import sys
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from scipy import signal

class pylith_gprs:
   '''
   Driver class for Pylith GPRS model
   '''
   def __init__(self,args):

       self.num_p = 1
       start_q = 100.0
       end_q = 400.0
       self.p_ = np.linspace(start_q,end_q,self.num_p)
       np.random.shuffle(self.p_) # shuffle apriori!!!

       self.num_tsteps = 115

       # too many parameters!!!
       self.window = 25
       self.stride = 1
       self.batch_size = 1
       self.hidden_size = 5

       self.num_features = 2

       self.num_layers = 1 

       self.consts={}
       self.consts["q"] = 1000 # dummy!!!

       self.args = args

       self.lstm_file = 'model_lstm.pickle'
       self.data_file = 'data.pickle'
 

   def solve(self):

       self.time_series() 

       if self.args.reduction:   
       # LSTM encoder-decoder!!!
          self.build_windowed_dataset()
          self.build_lstm()

       if self.args.bayesian:
       # bayesian!!!
          self.inference()      


   def time_series(self):

       self.t_ = np.linspace(1,self.num_tsteps,self.num_tsteps)
       t_appended =  np.zeros((self.num_p*self.num_tsteps,self.num_features))
       u_appended =  np.zeros((self.num_p*self.num_tsteps,self.num_features))
       u_appended_noise =  np.zeros((self.num_p*self.num_tsteps,self.num_features))
       
       count_q = 0
       for rate in self.p_:
          u = np.zeros((self.num_tsteps,2))
          directory = './vtk_plots' + '/%s' % int(rate)
          count_ = 0
          for file_name in sorted(os.listdir(directory)):
              filename = os.path.join(directory, file_name)
              if os.path.isfile(filename):
                  parser = parse_vtk(filename)
                  u[count_,0] = parser.get_surface_information("displacement")
                  u[count_,1] = u[count_,0]
                  count_ += 1

          temp = np.asarray(u).reshape(len(u),-1)
          u_noise = temp + 1.0*np.abs(temp)*np.random.randn(temp.shape[0],temp.shape[1]) #synthetic data

          u_ = u.reshape(-1,self.num_features)
          u_noise = u_noise.reshape(-1,self.num_features)
          start_ = count_q*self.num_tsteps
          end_ = start_ + self.num_tsteps

          t_appended[start_:end_,0] = self.t_[:]
          t_appended[start_:end_,1] = rate

          u_appended[start_:end_,0] = u_[:,0] 
          u_appended[start_:end_,1] = u_appended[start_:end_,0]

          u_appended_noise[start_:end_,0] = u_noise[:,0]
          u_appended_noise[start_:end_,1] = u_appended_noise[start_:end_,0]

          count_q += 1
  
       self.t_appended = t_appended
       self.u_appended = u_appended
 
       # pickle the data!!!
       save_object(u_appended_noise,self.data_file)

 
   def build_windowed_dataset(self):
    
       # build windowed dataset!!!
       t_ = self.t_appended.reshape(-1,self.num_features)
       var_ = self.u_appended.reshape(-1,self.num_features)
       self.num_samples_train, self.Ttrain, self.Ytrain = self.windowed_dataset(t_, var_) 


   def windowed_dataset(self,t, y):
    
       L = y.shape[0]
       num_samples = (L - self.window) // self.stride + 1
   
       Y = np.zeros([self.window, num_samples, self.num_features])     
       T = np.zeros([self.window, num_samples, self.num_features])     
      
       for ff in np.arange(self.num_features):

           for ii in np.arange(num_samples):

               start_x = self.stride * ii
               end_x = start_x + self.window

               self.end_x = end_x # end of final window may not be the end of data!!!
   
               Y[:, ii, ff] = y[start_x:end_x, ff] 
               T[:, ii, ff] = t[start_x:end_x, ff]
   
       return num_samples, T, Y


   def numpy_to_torch(self,Ttrain, Ytrain):
       '''
       convert numpy array to PyTorch tensor
       
       '''
       T_train_torch = torch.from_numpy(Ttrain).type(torch.Tensor)
       Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)
       
       return T_train_torch, Y_train_torch


   def build_lstm(self):
    
       # build rom!!!
       T_train, Y_train = self.numpy_to_torch(self.Ttrain, self.Ytrain)
       n_epochs = self.args.num_epochs
       input_tensor = T_train
       model_lstm = lstm_encoder_decoder.lstm_seq2seq(input_tensor.shape[2], self.hidden_size, self.num_layers, False)
       loss = model_lstm.train_model(input_tensor, Y_train, n_epochs, self.window, self.batch_size)
   
       # pickle the rom!!!
       save_object(model_lstm,self.lstm_file)

       # plot!!!
       self.plot(model_lstm, self.Ttrain, self.Ttrain, self.Ytrain, self.stride, self.window, 'Training', 'Reconstruction',)


   def plot(self,lstm_model, T_, X_, Y_, stride, window, dataset_type, objective):
     '''
     plot examples of the lstm encoder-decoder evaluated on the training/test data
     
     '''
     num_samples, num_p, p_, num_tsteps = self.num_samples_train, self.num_p, self.p_, self.num_tsteps 
     count_q = 0
     for q in p_:
   
         X = np.zeros([self.window,num_samples]) 
         Y = np.zeros([self.window,num_samples])     
         T = np.zeros([self.window,num_samples])     

         T__ = np.zeros([self.end_x])     
         X__ = np.zeros([self.end_x])     
         Y__ = np.zeros([self.end_x])     
  
         for ii in range(num_samples):
   
             train_plt = X_[:, ii, :]
             Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len = window)

             X[:,ii] = Y_[:, ii, 0]
             Y[:,ii] = Y_train_pred[:, 0]
             T[:,ii] = T_[:, ii, 0]

             if ii != 0:
               xy, ind1, ind2 = np.intersect1d(T[:,ii-1],T[:,ii],return_indices=True)
               end = 2*self.window - ind1.shape[0]

               T__[(ii-1)*stride+0:(ii-1)*stride+ind1[0]] = T[0:ind1[0],ii-1] 
               T__[(ii-1)*stride+ind1[0]:(ii-1)*stride+ind1[-1]] = (T[ind1[0]:ind1[-1],ii-1] + T[ind2[0]:ind2[-1],ii])/2 
               T__[(ii-1)*stride+ind1[-1]:(ii-1)*stride+end] = T[ind2[-1]:T[:,ii].shape[0],ii] 

               X__[(ii-1)*stride+0:(ii-1)*stride+ind1[0]] = X[0:ind1[0],ii-1] 
               X__[(ii-1)*stride+ind1[0]:(ii-1)*stride+ind1[-1]] = (X[ind1[0]:ind1[-1],ii-1] + X[ind2[0]:ind2[-1],ii])/2 
               X__[(ii-1)*stride+ind1[-1]:(ii-1)*stride+end] = X[ind2[-1]:X[:,ii].shape[0],ii] 

               Y__[(ii-1)*stride+0:(ii-1)*stride+ind1[0]] = Y[0:ind1[0],ii-1] 
               Y__[(ii-1)*stride+ind1[0]:(ii-1)*stride+ind1[-1]] = (Y[ind1[0]:ind1[-1],ii-1] + Y[ind2[0]:ind2[-1],ii])/2 
               Y__[(ii-1)*stride+ind1[-1]:(ii-1)*stride+end] = Y[ind2[-1]:Y[:,ii].shape[0],ii] 


         plt.rcParams.update({'font.size': 16})
         plt.figure()
         plt.plot(T__, X__, '-', color = (0.2, 0.42, 0.72), linewidth = 1.0, markersize = 1.0, label = 'Target')
         plt.plot(T__, Y__, '-', color = (0.76, 0.01, 0.01), linewidth = 1.0, markersize = 1.0, label = '%s' % objective)
         plt.xlabel('Time stamp')
         plt.ylabel('Disp $(m)$')
         plt.legend(frameon=False)
         plt.suptitle('%s data set for q=%s MSCF/day' % (dataset_type,q), x = 0.445, y = 1.)
         plt.tight_layout()
         plt.savefig('plots/%s_%s.png' % (q,dataset_type))
   
         count_q += 1
   
         del X,Y,T
   

   def plot_results(self,lstm_model, T_, X_, Y_, stride, window, dataset_type, objective):
     '''
     plot examples of the lstm encoder-decoder evaluated on the training/test data
     
     '''
     num_samples, num_p, p_, num_tsteps = self.num_samples_train, self.num_p, self.p_, self.num_tsteps 
     count_q = 0
     for q in p_:
   
         X = np.zeros([int(num_samples*window/num_p)]) 
         Y = np.zeros([int(num_samples*window/num_p)])     
         T = np.zeros([int(num_samples*window/num_p)])     
   
         num_samples_per_q = int(num_samples/num_p) 
   
         for ii in range(num_samples_per_q):
   
             start = ii*stride
             end = start + window
   
             train_plt = X_[:, count_q*num_samples_per_q+ii, :]
             Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len = window)
   
             X[start:end] = Y_[:, count_q*num_samples_per_q+ii, 0]
             Y[start:end] = Y_train_pred[:, 0]
             T[start:end] = T_[:, count_q*num_samples_per_q+ii, 0]
   
         plt.rcParams.update({'font.size': 16})
         plt.figure()
         plt.plot(T, X, '-', color = (0.2, 0.42, 0.72), linewidth = 1.0, markersize = 1.0, label = 'Target')
         plt.plot(T, Y, '-', color = (0.76, 0.01, 0.01), linewidth = 1.0, markersize = 1.0, label = '%s' % objective)
         plt.xlabel('Time stamp')
         plt.ylabel('Disp $(m)$')
         plt.legend(frameon=False)
         plt.suptitle('%s data set for q=%s MSCF/day' % (dataset_type,q), x = 0.445, y = 1.)
         plt.tight_layout()
         plt.savefig('plots/%s_%s.png' % (q,dataset_type))
   
         count_q += 1
   
         del X,Y,T
   

   def inference(self):

       # load reduced order model!!!
       model_lstm = load_object(self.lstm_file)

       # load data!!!
       u_appended_noise = load_object(self.data_file)
   
       num_p = self.num_p
       num_tsteps = self.num_tsteps
       p_ = self.p_
   
       for ii in range(0,num_p):
   
           noisy = u_appended_noise[ii*num_tsteps:ii*num_tsteps+num_tsteps,0]
           noisy = noisy.reshape(1, num_tsteps)
       
           q = p_[ii]
           print('--- q is %s ---' % q)
   
           qstart={"q":1} # Initial guess of 1!!!
           qpriors={"q":["Uniform",1,500]}
   
           nsamples = self.args.num_samples
           nburn = nsamples/2
           
           problem_type = 'rom'
           MCMCobj2=MCMC(self,qpriors=qpriors,nsamples=nsamples,nburn=nburn,data=noisy,problem_type=problem_type,lstm_model=model_lstm,qstart=qstart,adapt_interval=100,verbose=True)
           qparams2=MCMCobj2.sample() # run the Bayesian/MCMC algorithm
           self.plot_dist(qparams2,q)
   

   def rom_evaluate(self,params,lstm_model):
        
       for arg in params.keys():
           self.consts[arg] = params[arg]

       num_steps = self.num_tsteps 
       window = self.window
   
       # Create arrays to store trajectory
       t = np.zeros((num_steps,2)) 
       acc = np.zeros((num_steps,2)) 
   
       t[:,0] = self.t_[:]
       t[:,1] = self.consts["q"]
  
       num_steps_ = int(num_steps/window) 
       train_plt = np.zeros((window,2)) 
    
       for ii in range(num_steps_):
   
           start = ii*window
           end = start + window
   
           train_plt[0:window,0] = t[start:end,0]
           train_plt[0:window,1] = t[start:end,1]
           Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len = window)
           acc[start:end,0] = Y_train_pred[:, 0]
           acc[start:end,1] = Y_train_pred[:, 0]
 
       return t, acc


   def plot_dist(self, qparams, q):

       plt.rcParams.update({'font.size': 14})
       n_rows = 1
       n_columns = 2
       gridspec = {'width_ratios': [0.7, 0.15], 'wspace': 0.15}
       fig, ax = plt.subplots(n_rows, n_columns, gridspec_kw=gridspec)
       fig.suptitle('q=%s MSCF/day' % q)
       ax[0].plot(qparams[0,:], 'b-', linewidth=1.0)
       ylims = ax[0].get_ylim()
       x = np.linspace(ylims[0], ylims[1], 1000)
       kde = gaussian_kde(qparams[0,:])
       ax[1].plot(kde.pdf(x), x, 'b-')
       max_val = x[kde.pdf(x).argmax()]
       ax[1].plot(kde.pdf(x)[kde.pdf(x).argmax()],max_val, 'ro')
       ax[1].annotate(str(round(max_val,2)),xy=(1.05*kde.pdf(x)[kde.pdf(x).argmax()],max_val),size=14)
       ax[1].fill_betweenx(x, kde.pdf(x), np.zeros(x.shape), alpha=0.3)
       ax[1].set_xlim(0, None)
       ax[0].set_ylabel('q')
       ax[0].set_xlim(0, qparams.shape[1]) 
       ax[0].set_xlabel('Sample number')
       ax[1].set_xlabel('Prob. density')
       ax[1].get_yaxis().set_visible(False)
       ax[1].get_xaxis().set_visible(True)
       ax[1].get_xaxis().set_ticks([])
       fig.savefig('./plots/pylith_gprs_burn_%s_%s_.png' % (q,self.args.num_samples))
