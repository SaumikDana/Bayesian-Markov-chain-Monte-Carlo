__author__ = "Saumik Dana"
__purpose__ = "Demonstrate Bayesian inference using RSF model"

from rsf import rsf
from RateStateModel import RateStateModel
import matplotlib.pyplot as plt

def run_simulation(qpriors, format, nsamples):
    # Inference problem setup
    problem = rsf(number_slip_values=5,
                  lowest_slip_value=100.,
                  largest_slip_value=1000.,
                  qstart=10.,
                  qpriors=qpriors)

    # RSF model setup
    problem.model = RateStateModel(number_time_steps=500)

    # Generate the time series for the RSF model
    data = problem.generate_time_series()

    # Set the data format and perform Bayesian inference
    problem.format = format  # Ensuring the format is used in the problem
    time_taken = problem.inference(data, nsamples=nsamples)

    return time_taken

def visualize_data():
    plt.show()
