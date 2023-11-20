from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from rsf import rsf
from ratestatemodel import RateStateModel
import matplotlib.pyplot as plt

app = FastAPI()

class SimulationParams(BaseModel):
    qpriors: List
    format: str  # Including the format parameter
    nsamples: int

@app.post("/run-simulation")
def run_simulation_endpoint(params: SimulationParams):
    if params.format not in ["json", "mysql"]:
        return {"error": "Invalid format specified"}

    time_taken = run_simulation(params.qpriors, params.format, params.nsamples)
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

def run_simulation(qpriors, format, nsamples):
    # Inference problem setup
    problem = rsf(number_slip_values=NUMBER_SLIP_VALUES,
                  lowest_slip_value=LOWEST_SLIP_VALUE,
                  largest_slip_value=LARGEST_SLIP_VALUE,
                  qstart=QSTART,
                  qpriors=qpriors)
    problem.model = RateStateModel(number_time_steps=NUMBER_TIME_STEPS)

    # Generate the time series for the RSF model
    data = problem.generate_time_series()

    # Set the data format and perform Bayesian inference
    problem.format = format  # Ensuring the format is used in the problem
    time_taken = problem.inference(data, nsamples=nsamples)

    return time_taken

def visualize_data():
    plt.show()
