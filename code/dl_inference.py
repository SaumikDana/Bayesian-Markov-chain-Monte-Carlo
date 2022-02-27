from rsf import rsf
from pylith_gprs import pylith_gprs
from inference import rsf_inference,rsf_inference_no_rom,pylith_gprs_inference
import numpy as np
import matplotlib.pyplot as plt

    
def main(problem,rom,bayesian):

    if problem == 'rsf':
    # rsf problem!!!
       problem_ = rsf()
       t_appended, acc_appended, file2 = problem_.time_series() 

       if rom:
       # LSTM encoder-decoder!!!
         file1 = problem_.build_lstm(t_appended,acc_appended)

       if bayesian:
       # bayesian!!!
          if rom:
             rsf_inference(file1,file2,problem_.num_p,problem_.num_tsteps,problem_.p_,problem_.model)      
          else:
             rsf_inference_no_rom(file1,file2,problem_.num_p,problem_.num_tsteps,problem_.p_,problem_.model)      

    elif problem == 'pylith_gprs':
    # pylith-gprs problem!!!
       problem_ = pylith_gprs()
       t_appended, u_appended, u_appended_noise = problem_.time_series() 

       if rom:   
       # LSTM encoder-decoder!!!
          problem_.build_lstm(t_appended,u_appended)

       if bayesian:
       # bayesian!!!
          problem_.inference(u_appended_noise)      

    # Close it out!!!
    plt.show()
    plt.close('all')


# Driver code!!!
if __name__ == '__main__':

#    problem = 'rsf'
    problem = 'pylith_gprs'
    rom = False
#    rom = True
#    bayesian = False
    bayesian = True
    main(problem,rom,bayesian)




