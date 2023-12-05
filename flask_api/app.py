from flask import Flask, request
from flask_restful import Api, Resource
from pydantic import BaseModel
from typing import List

import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from source_python.RSF import RSF
from source_python.RateStateModel import RateStateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

app = Flask(__name__)
api = Api(app)

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
        params = InferenceParams(**request.json)
        time_taken = run_inference(params.nsamples)
        return {"time_taken": time_taken}

class Visualize(Resource):
    def get(self):
        visualize_data()
        return {"message": "Data visualization initiated"}

# Constants
QSTART = 1000.
QPRIORS = ["Uniform", 0., 10000.]
NUMBER_TIME_STEPS = 500

def run_simulation(nslips, lowest, largest):
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

    return

def run_inference(nsamples):
    # Load the problem object using pickle
    with open('problem_data.pkl', 'rb') as file:  # Note the 'rb' mode for binary read
        problem = pickle.load(file)

    # Delete the pickle file after loading
    os.remove('problem_data.pkl')

    # Json format for saving and loading Objects
    problem.format = 'json'
    
    # Perform inference
    time_taken = problem.inference(nsamples=nsamples)

    return time_taken

def visualize_data():
    plt.show()

# Register the routes
api.add_resource(Root, '/')
api.add_resource(RunSimulation, '/run-simulation')
api.add_resource(RunInference, '/run-inference')
api.add_resource(Visualize, '/visualize')

# Core functions (run_simulation, run_inference, visualize_data) remain the same

if __name__ == '__main__':
    app.run(debug=True)
