from rsf import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # rsf problem constructor
    problem = rsf(number_slip_values=5)
    
    # Generate the time series for the RSF model
    problem.time_series() 

    # Bayesian inference
    problem.solve()

    # Close it out
    plt.show()
    plt.close('all')

