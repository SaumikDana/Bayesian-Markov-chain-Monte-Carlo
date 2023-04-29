from rsf import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Rate State model
    problem       = rsf(number_slip_values=5)
    problem.model = RateStateModel()

    # Generate the time series for the RSF model
    problem.time_series()

    # Flag for problem type
    reduction = True 

    if reduction:
        # Use LSTM encoder-decoder for dimensionality reduction
        problem.build_lstm(epochs=20, num_layers=1)

    # # Perform Bayesian inference
    # problem.qstart  = 100.
    # problem.qpriors = ["Uniform",0.,10000.]
    # problem.inference(nsamples=500,reduction=reduction)      

    # Close it out
    plt.show()
    plt.close('all')

