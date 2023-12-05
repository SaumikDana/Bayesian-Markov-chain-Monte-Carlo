from fastapi import FastAPI
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

app = FastAPI()

class SimulationParams(BaseModel):
    nslips: int
    lowest: float
    largest: float

class InferenceParams(BaseModel):
    nsamples: int

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI application!"}

@app.post("/run-simulation")
def run_simulation_endpoint(params: SimulationParams):
    run_simulation(params.nslips, params.lowest, params.largest)
    return {"message": "Forward simulation Complete! Data generated"}

@app.post("/run-inference")
def run_inference_endpoint(params: InferenceParams):
    time_taken = run_inference(params.nsamples)
    return {"time_taken": time_taken}

@app.get("/visualize")
def visualize_endpoint():
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
