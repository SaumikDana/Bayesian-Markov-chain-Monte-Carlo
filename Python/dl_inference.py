from rsf import rsf
import matplotlib.pyplot as plt
from RateStateModel import RateStateModel

if __name__ == '__main__':

    # Inference problem
    problem = rsf(number_slip_values=5,lowest_slip_value=100.,largest_slip_value=1000.,plotfigs=True)

    # RSF model
    problem.model = RateStateModel(number_time_steps=500)

    # Data format
    problem.format = 'sql'

    # Generate the time series for the RSF model
    data = problem.generate_time_series()

    if problem.format == 'json':
        from json_save_load import save_object, load_object
        # Define file names for saving and loading data and LSTM model
        problem.lstm_file  = 'model_lstm.json'
        problem.data_file  = 'data.json'
    elif problem.format == 'sql':
        from sql_save_load import save_object, load_object
        # Define file names for saving and loading data and LSTM model
        problem.lstm_file  = 'model_lstm.db'
        problem.data_file  = 'data.db'

    # Save the data & then load the saved data file
    save_object(data,problem.data_file)
    data = load_object(problem.data_file)  

    # Perform Bayesian inference
    problem.qstart  = 10.
    problem.qpriors = ["Uniform",0.,10000.]
    problem.inference(data,nsamples=500)      

    # Close it out
    plt.show()
    plt.close('all')

