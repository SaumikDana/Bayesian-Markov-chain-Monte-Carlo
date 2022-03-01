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

class pylith_gprs:
   '''
   Driver class for Pylith GPRS model
   '''
   def __init__(self,args):

       self.num_p = 7
       start_q = 100.0
       end_q = 400.0
       self.p_ = np.linspace(start_q,end_q,self.num_p)

       self.num_tsteps = 115
       self.t_ = np.linspace(1,self.num_tsteps,self.num_tsteps)

       self.window = 23 # make sure num_tsteps is exact multiple of window!!!
       self.stride = self.window
       self.num_features = 2
       self.hidden_size = self.window/2
       self.batch_size = 1
       self.num_layers = 1 

       self.consts={}
       self.consts["q"] = 1000 # dummy!!!

       self.args = args


   def solve(self):

       t_appended, u_appended, u_appended_noise = self.time_series() 

       if self.args.reduction:   
       # LSTM encoder-decoder!!!
          self.build_lstm(t_appended,u_appended)

       if self.args.bayesian:
       # bayesian!!!
          self.inference(u_appended_noise)      


   def time_series(self):

       num_p = self.num_p
       p_ = self.p_
       t_ = self.t_
       num_tsteps = self.num_tsteps
       num_features = self.num_features
       num_features = self.num_features

       t_appended =  np.zeros((num_p*num_tsteps,num_features))
       u_appended =  np.zeros((num_p*num_tsteps,num_features))
       u_appended_noise =  np.zeros((num_p*num_tsteps,num_features))
       
       u = np.zeros((num_tsteps,2))
       count_q = 0
       for rate in p_:
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

          u_ = u.reshape(-1,num_features)
          u_noise = u_noise.reshape(-1,num_features)
          start_ = count_q*num_tsteps
          end_ = start_ + num_tsteps

          t_appended[start_:end_,0] = t_[:]
          t_appended[start_:end_,1] = rate

          u_appended[start_:end_,0] = u_[:,0]
          u_appended[start_:end_,1] = u_appended[start_:end_,0]

          u_appended_noise[start_:end_,0] = u_noise[:,0]
          u_appended_noise[start_:end_,1] = u_appended_noise[start_:end_,0]

          count_q += 1
   
       return t_appended, u_appended, u_appended_noise

 
   def build_lstm(self, t_appended, var_appended):
    
       # build rom!!!
       t_ = t_appended.reshape(-1,self.num_features); var_ = var_appended.reshape(-1,self.num_features)
       num_samples_train, Ttrain, Ytrain = generate_dataset.windowed_dataset(t_, var_, self.window, self.stride, self.num_features) 
       T_train, Y_train = generate_dataset.numpy_to_torch(Ttrain, Ytrain)
       n_epochs = self.args.num_epochs
       input_tensor = T_train
       model_lstm = lstm_encoder_decoder.lstm_seq2seq(input_tensor.shape[2], self.hidden_size, self.num_layers, False)
       loss = model_lstm.train_model(input_tensor, Y_train, n_epochs, self.window, self.batch_size)
   
       # pickle the rom!!!
       my_file = Path(os.getcwd()+'/model_lstm.pickle')
       if not my_file.is_file() or self.args.reduction:
          save_object(model_lstm,"model_lstm.pickle") 

       # plot!!!
       self.plot_results(model_lstm, Ttrain, Ttrain, Ytrain, self.stride, self.window, 'Training', 'Reconstruction', num_samples_train, self.num_p, self.p_, self.num_tsteps)


   def plot_results(self,lstm_model, T_, X_, Y_, stride, window, dataset_type, objective, num_samples, num_p, p_, num_tsteps):
     '''
     plot examples of the lstm encoder-decoder evaluated on the training/test data
     
     '''
     Y_return = np.zeros([int(num_samples*window)])     
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
             Y_return[count_q*num_tsteps+start:count_q*num_tsteps+end] = Y_train_pred[:, 0]
   
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
   

   def inference(self,u_appended_noise):
       # load objects!!!
       model_lstm = load_object('model_lstm.pickle')  # ROM
   
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
           std_MCMC2 = MCMCobj2.std2
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
