from rsf import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Rate State model
    problem = rsf(number_slip_values=5,lowest_slip_value=100.,largest_slip_value=1000.,plotfigs=True)

    # Generate the time series for the RSF model
    problem.generate_time_series()

    # Perform Bayesian inference
    problem.qstart  = 100.
    problem.qpriors = ["Uniform",0.,10000.]
    problem.inference(nsamples=500)      

    # Close it out
    plt.show()
    plt.close('all')

