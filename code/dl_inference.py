from rsf import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Rsf problem constructor
    problem       = rsf(number_slip_values=5)

    # Rate State model
    problem.model = RateStateModel()

    # Generate the time series for the RSF model
    problem.time_series()

    # flags for problem type
    reduction = False # Use ROM

    if reduction:
        # Use LSTM encoder-decoder for dimensionality reduction
        problem.build_lstm()
        # Perform RSF inference with LSTM encoder-decoder
        problem.rsf_inference(nsamples=100)      
    else:
        # Perform RSF inference without dimensionality reduction
        problem.inference_full(nsamples=500)

    # Close it out
    plt.show()
    plt.close('all')

