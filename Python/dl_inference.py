from rsf import rsf
import matplotlib.pyplot as plt
from RateStateModel import RateStateModel

if __name__ == '__main__':

    # Inference problem
    qpriors = ["Uniform", 0., 10000.]
    
    problem = rsf(number_slip_values=5,
                  lowest_slip_value=100.,
                  largest_slip_value=1000.,
                  qstart=10.,
                  qpriors=qpriors)

    # RSF model
    problem.model = RateStateModel(number_time_steps=500)

    # Generate the time series for the RSF model
    data = problem.generate_time_series()

    # Perform Bayesian inference with mysql
    problem.format = 'mysql' # Data format
    mysql_time = problem.inference(data,nsamples=500)      

    # Perform Bayesian inference with json
    problem.format = 'json' # Data format
    json_time = problem.inference(data,nsamples=500)      

    print(f'\n Run Time For Inference: {mysql_time} & {json_time} seconds with mysql & json respectively \n')

    # Close it out
    plt.show()
    plt.close('all')

