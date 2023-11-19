__author__ = "Saumik Dana"
__purpose__ = "Demonstrate Bayesian inference using RSF model"

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from simulation import run_simulation, visualize_data

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
