from rsf import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Rsf problem constructor
    problem = rsf(number_slip_values=5)
    
    # Generate the time series for the RSF model
    problem.plotfigs = False
    problem.time_series() 

    # flags for problem type
    bayesian = True # Use Bayesian inference to estimate the critical slip distance
    reduction = False # Use ROM

    if bayesian and reduction:
        # Use LSTM encoder-decoder for dimensionality reduction
        problem.build_lstm()
        # Perform RSF inference with LSTM encoder-decoder
        problem.rsf_inference()      
    elif bayesian and not reduction:
        # Perform RSF inference without dimensionality reduction
        problem.inference_full(nsamples=500)

    # Close it out
    plt.show()
    plt.close('all')

