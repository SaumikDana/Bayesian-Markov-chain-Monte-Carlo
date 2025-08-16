import setup_path
from src.imports import *


# Constants
NUMBER_SLIP_VALUES = 5
LOWEST_SLIP_VALUE = 100.
LARGEST_SLIP_VALUE = 5000.
QSTART = 1000.
QPRIORS = ["Uniform", 0., 10000.]
NUMBER_TIME_STEPS = 500
NSAMPLES = 500

def setup_problem():
    """
    Set up the RSF problem with specified parameters.

    Returns:
        problem: Configured RSF problem instance.
    """
    problem       = RSF(number_slip_values=NUMBER_SLIP_VALUES,
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

    json_time = perform_inference(problem, 'json', NSAMPLES)

    plt.show()
    plt.close('all')

if __name__ == '__main__':
    main()


