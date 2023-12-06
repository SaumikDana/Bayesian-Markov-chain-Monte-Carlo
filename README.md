# Bayesian MCMC in Rate State Models

## Introduction
`dl_inference` is a tool that implements a Rate State model and performs Bayesian inference using MCMC sampling, designed for Python and C++ environments.

## Running the Source code Directly
### Python Version
1. Install dependencies.
2. Navigate to source_c++ folder
3. Run Script: `python3 -m source_python.dl_inference`

### C++ Version
1. Install dependencies (boost, gsl, eigen)
2. Navigate to source_c++ folder
3. Compile: `g++ -std=c++17 -o executable dl_inference.cpp RateStateModel.cpp MCMC.cpp [...]`
4. Run executable

### Features
- Rate State Model Initialization with customizable parameters.
- Time Series Generation and Bayesian Inference using MCMC.
- Visualization with `matplotlib`.

## Running FastAPI
1. Navigate to fastapi folder
2. Run the main script: `uvicorn main:app --reload`

## Running Flask
1. Navigate to flask folder
2. `export FLASK_APP=application.py`
3. `flask run`
4. On a separate terminal: `python3 flask_api_tester.py`. This will fire up the source code in terminal on which flask is running

## Paper Summary
Based on "Arriving at estimates of a rate and state fault friction model parameter using Bayesian inference and Markov chain Monte Carlo", https://www.sciencedirect.com/science/article/pii/S266654412200003X

## Contact
For support or collaboration: dana.spk5@gmail.com

---
Copyright © [Saumik Dana]
---