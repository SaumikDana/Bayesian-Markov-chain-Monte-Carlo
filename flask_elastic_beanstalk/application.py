import logging
from flask import Flask, request
from flask_restful import Api, Resource
from pydantic import BaseModel

from source_python.RSF import RSF
from source_python.RateStateModel import RateStateModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

application = Flask(__name__)
api = Api(application)

class SimulationParams(BaseModel):
    nslips: int
    lowest: float
    largest: float

class InferenceParams(BaseModel):
    nsamples: int

class Root(Resource):
    def get(self):
        return {"message": "Welcome to my Flask application!"}

class RunSimulation(Resource):
    def post(self):
        params = SimulationParams(**request.json)
        run_simulation(params.nslips, params.lowest, params.largest)
        return {"message": "Forward simulation Complete! Data generated"}

class RunInference(Resource):
    def post(self):
        logging.info("Received request for /run-inference with data: %s", request.json)
        try:
            params = InferenceParams(**request.json)
            time_taken = run_inference(params.nsamples)
            logging.info("Inference completed successfully in %s seconds", time_taken)
            return {"time_taken": time_taken}
        except Exception as e:
            logging.error("Error during /run-inference: %s", str(e))
            return {"error": str(e)}, 500

class Visualize(Resource):
    def get(self):
        visualize_data()
        return {"message": "Data visualization initiated"}

# Constants
QSTART = 1000.
QPRIORS = ["Uniform", 0., 10000.]
NUMBER_TIME_STEPS = 500

def run_simulation(nslips, lowest, largest):
    logging.info(
        "Starting run_simulation with nslips: %d, lowest: %f, largest: %f", 
        nslips, lowest, largest
    )
    try:
        # Problem setup
        problem = RSF(number_slip_values=nslips,
                    lowest_slip_value=lowest,
                    largest_slip_value=largest,
                    qstart=QSTART,
                    qpriors=QPRIORS
                    )
        problem.model = RateStateModel(number_time_steps=NUMBER_TIME_STEPS)
        problem.data = problem.generate_time_series()

        # Serialize and save the problem object using pickle
        with open('problem_data.pkl', 'wb') as file:  # Note the 'wb' mode for binary write
            pickle.dump(problem, file)
        logging.info("Simulation completed successfully")
    except Exception as e:
        logging.error("Error in run_simulation: %s", str(e))
        raise
        
    return

def run_inference(nsamples):
    try:
        logging.info("Starting run_inference with nsamples: %d", nsamples)
        # Load the problem object using pickle
        with open('problem_data.pkl', 'rb') as file:  # Note the 'rb' mode for binary read
            problem = pickle.load(file)

        # Delete the pickle file after loading
        os.remove('problem_data.pkl')

        # Json format for saving and loading Objects
        problem.format = 'json'
        
        # Perform inference
        time_taken = problem.inference(nsamples=nsamples)
        logging.info("Inference processing completed")
        return time_taken
    except Exception as e:
        logging.error("Error in run_inference: %s", str(e))
        raise
        
def visualize_data():
    plt.show()

# Register the routes
api.add_resource(Root, '/')
api.add_resource(RunSimulation, '/run-simulation')
api.add_resource(RunInference, '/run-inference')
api.add_resource(Visualize, '/visualize')

# Condition to check if running locally or on Elastic Beanstalk
if __name__ == '__main__':
    # Only run the application manually if it's not run by a WSGI server like on Elastic Beanstalk
    port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ["true", "1", "t"]
    application.run(host=host, port=port, debug=debug)