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

app = FastAPI()

class SimulationParams(BaseModel):
    format: str  # Including the format parameter
    nsamples: int

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI application!"}

@app.post("/run-simulation")
def run_simulation_endpoint(params: SimulationParams):
    if params.format not in ["json", "mysql"]:
        return {"error": "Invalid format specified"}

    time_taken = run_simulation(params.format, params.nsamples)
    return {"time_taken": time_taken}

@app.get("/visualize")
def visualize_endpoint():
    visualize_data()
    return {"message": "Data visualization initiated"}

NUMBER_SLIP_VALUES = 5
LOWEST_SLIP_VALUE = 100.
LARGEST_SLIP_VALUE = 1000.
QSTART = 10.
NUMBER_TIME_STEPS = 500
QPRIORS = ["Uniform", 0., 10000.]

def run_simulation(format, nsamples):
    # Inference problem setup
    problem       = RSF(number_slip_values=NUMBER_SLIP_VALUES,
                        lowest_slip_value=LOWEST_SLIP_VALUE,
                        largest_slip_value=LARGEST_SLIP_VALUE,
                        qstart=QSTART,
                        qpriors=QPRIORS
                        )
    problem.model = RateStateModel(number_time_steps=NUMBER_TIME_STEPS)

    # Generate the time series for the RSF model
    data = problem.generate_time_series()

    # Set the data format and perform Bayesian inference
    problem.format = format  # Ensuring the format is used in the problem
    time_taken = problem.inference(data, nsamples=nsamples)

    return time_taken

def visualize_data():
    plt.show()
