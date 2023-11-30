from rsf import rsf
import matplotlib.pyplot as plt
from ratestatemodel import RateStateModel

# Constants
NUMBER_SLIP_VALUES = 5
LOWEST_SLIP_VALUE = 100.
LARGEST_SLIP_VALUE = 1000.
QSTART = 10.
QPRIORS = ["Uniform", 0., 10000.]
NUMBER_TIME_STEPS = 500
NSAMPLES = 500

def setup_problem():
    """
    Set up the RSF problem with specified parameters.

    Returns:
        problem: Configured RSF problem instance.
    """
    problem       = rsf(number_slip_values=NUMBER_SLIP_VALUES,
                        lowest_slip_value=LOWEST_SLIP_VALUE,
                        largest_slip_value=LARGEST_SLIP_VALUE,
                        qstart=QSTART,
                        qpriors=QPRIORS
                        )
    problem.model = RateStateModel(number_time_steps=NUMBER_TIME_STEPS)
    problem.data  = problem.generate_time_series()
    return problem

def perform_inference(problem, data_format, nsamples):
    """
    Perform Bayesian inference on the problem.

    Args:
        problem: RSF problem instance.
        data_format: Format of the data ('mysql' or 'json').
        nsamples: Number of samples for inference.

    Returns:
        Time taken for the inference process.
    """
    problem.format = data_format
    return problem.inference(nsamples=nsamples)

def main():
    """
    Main function to run the Bayesian inference problem.
    """
    problem = setup_problem()

    mysql_time = perform_inference(problem, 'mysql', NSAMPLES)
    json_time = perform_inference(problem, 'json', NSAMPLES)

    print(f'\nRun Time For Inference: {mysql_time} & {json_time} seconds with mysql & json respectively\n')

    plt.show()
    plt.close('all')

if __name__ == '__main__':
    main()


