from rsf import rsf
import matplotlib.pyplot as plt
from json_save_load import save_object, load_object
from RateStateModel import RateStateModel

if __name__ == '__main__':

    # Inference problem
    problem = rsf(number_slip_values=5,lowest_slip_value=100.,largest_slip_value=1000.,plotfigs=True)

    # RSF model
    problem.model = RateStateModel(number_time_steps=500)

    # Generate the time series for the RSF model
    data = problem.generate_time_series()

    # Save the data using json
    save_object(data,problem.data_file)

    # Perform Bayesian inference
    problem.qstart  = 100.
    problem.qpriors = ["Uniform",0.,10000.]
    data = load_object(problem.data_file)  # Load a saved data file
    problem.inference(data,nsamples=500)      

    # Close it out
    plt.show()
    plt.close('all')

