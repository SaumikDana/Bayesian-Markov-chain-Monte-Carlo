from RateStateModel import RateStateModel
import numpy as np
import math

class rsf:
   '''
   Driver class for RSF model
   '''
   def __init__(self):

       self.num_dc = 2
       start_dc = 1.0
       end_dc = 1000.0
       self.dc_ = np.logspace(math.log10(start_dc),math.log10(end_dc),self.num_dc)

       t_start = 0.0
       t_end = 50.0
       self.num_tsteps = 500

       self.window = 25 # make sure num_tsteps is exact multiple of window!!!
       self.stride = 25
       self.num_features = 2

       self.model = RateStateModel(t_start, t_end, num_tsteps = self.num_tsteps, window = self.window) # forward model

   def time_series(self):

       num_dc = self.num_dc
       dc_ = self.dc_
       num_tsteps = self.num_tsteps
       num_features = self.num_features
       model = self.model
       num_features = self.num_features

       t_appended =  np.zeros((num_dc*num_tsteps,num_features))
       acc_appended =  np.zeros((num_dc*num_tsteps,num_features))
       acc_appended_noise = np.zeros((num_dc*num_tsteps,num_features))
       
       count_dc = 0
       for dc in dc_:
          model.set_dc(dc)
          t, acc, acc_noise = model.evaluate(model.consts) # noisy data
          t_ = t.reshape(-1,num_features); acc_ = acc.reshape(-1,num_features)
          start_ = count_dc*num_tsteps; end_ = start_ + num_tsteps
          t_appended[start_:end_,0] = t[:,0]; t_appended[start_:end_,1] = dc
          acc_appended[start_:end_,0] = acc[:,0]; acc_appended[start_:end_,1] = acc[:,1]
          acc_appended_noise[start_:end_,0] = acc_noise[:,0]; acc_appended_noise[start_:end_,1] = acc_noise[:,1]
          count_dc += 1
   
       return t_appended, acc_appended, acc_appended_noise
    

