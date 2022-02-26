from RateStateModel import RateStateModel
import numpy as np
import math

class rsf:
   '''
   Driver class for RSF model
   '''
   def __init__(self):

       self.num_p = 1
       start_dc = 1000.0
       end_dc = 1000.0
       self.p_ = np.logspace(math.log10(start_dc),math.log10(end_dc),self.num_p)

       t_start = 0.0
       t_end = 50.0
       self.num_tsteps = 500

       self.window = 25 # make sure num_tsteps is exact multiple of window!!!
       self.stride = 25
       self.num_features = 2

       self.model = RateStateModel(t_start, t_end, num_tsteps = self.num_tsteps, window = self.window) # forward model

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

       # save objects!!!
       my_file = Path(os.getcwd()+'/acc_appended_noise.pickle')
       if not my_file.is_file():
          save_object(acc_appended_noise,"acc_appended_noise.pickle")
   
       return t_appended, acc_appended, acc_appended_noise

    
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
   
   
   
