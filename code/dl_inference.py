from rsf import rsf
from pylith_gprs import pylith_gprs
from inference import rsf_inference,rsf_inference_no_rom,pylith_gprs_inference
import numpy as np
import matplotlib.pyplot as plt
import argparse

    
def main(args):

    if args.problem == 'rsf':
    # rsf problem!!!
       problem_ = rsf()
       t_appended, acc_appended, file2 = problem_.time_series() 

       if args.reduction:
       # LSTM encoder-decoder!!!
         file1 = problem_.build_lstm(t_appended,acc_appended)

       if args.bayesian:
       # bayesian!!!
          if rom:
             rsf_inference(file1,file2,problem_.num_p,problem_.num_tsteps,problem_.p_,problem_.model)      
          else:
             rsf_inference_no_rom(file1,file2,problem_.num_p,problem_.num_tsteps,problem_.p_,problem_.model)      

    elif args.problem == 'coupled':
    # pylith-gprs problem!!!
       problem_ = pylith_gprs(args)
       t_appended, u_appended, u_appended_noise = problem_.time_series() 

       if args.reduction:   
       # LSTM encoder-decoder!!!
          problem_.build_lstm(t_appended,u_appended)

       if args.bayesian:
       # bayesian!!!
          problem_.inference(u_appended_noise)      

    # Close it out!!!
    plt.show()
    plt.close('all')


# Driver code!!!
if __name__ == '__main__':

    #arguments!!!
    #Usage: python dl_inference.py -problem coupled -nepochs 100 -nsamples 100 --bayesian
    parser = argparse.ArgumentParser()
    parser.add_argument('-problem', dest='problem', type=str, help="Problem type")
    parser.add_argument('-epochs',dest='num_epochs', type=int, help="Number of Epochs")
    parser.add_argument('-samples',dest='num_samples', type=int, help="Number of Samples")
    parser.add_argument('--reduction', dest='reduction', action='store_true')
    parser.add_argument('--bayesian', dest='bayesian', action='store_true')

    args = parser.parse_args()

    main(args)




