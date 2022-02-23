from parse_vtk import parse_vtk
import numpy as np

class pylith_gprs:
   '''
   Driver class for Pylith GPRS model
   '''
   def __init__(self):

       self.num_q = 13
       start_q = 100.0
       end_q = 400.0
       self.q_ = np.linspace(start_q,end_q,self.num_q)

       self.num_tsteps = 19

       self.window = 1 # make sure num_tsteps is exact multiple of window!!!
       self.stride = 1
       self.num_features = 2

   def time_series(self):

       num_q = self.num_q
       q_ = self.q_
       num_tsteps = self.num_tsteps
       num_features = self.num_features
       model = self.model
       num_features = self.num_features

       t_appended =  np.zeros((num_q*num_tsteps,num_features))
       ux_appended =  np.zeros((num_q*num_tsteps,num_features))
       uy_appended = np.zeros((num_q*num_tsteps,num_features))
       
       ux, uy = np.zeros((num_tsteps,2)), np.zeros((num_tsteps,2))
       count_q = 0
       for rate in q_:
          directory = './vtk_plots' + '/%s' % rate
          count_ = 0
          for file_name in sorted(os.listdir(directory)):
              filename = os.path.join(directory, file_name)
              if os.path.isfile(filename):
                  parser = parse_vtk(filename)
                  ux[count_,0], uy[count_,0] = parser.get_surface_information("displacement")
                  ux[count_,1] = ux[count_,0]
                  uy[count_,1] = uy[count_,0]
                  count_ += 1

          ux_ = ux.reshape(-1,num_features)
          uy_ = uy.reshape(-1,num_features)
          start_ = count_q*num_tsteps
          end_ = start_ + num_tsteps
          t_appended[start_:end_,0] = q_[:,0]
          t_appended[start_:end_,1] = rate
          ux_appended[start_:end_,0] = ux_[:,0]
          ux_appended[start_:end_,1] = ux_[:,1]
          uy_appended[start_:end_,0] = uy_[:,0]
          uy_appended[start_:end_,1] = uy_[:,1]
          count_q += 1
   
       return t_appended, ux_appended, uy_appended
    

