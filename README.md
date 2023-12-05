# Bayesian MCMC in Rate State Models

## Introduction
`dl_inference` is a tool that implements a Rate State model and performs Bayesian inference using MCMC sampling, designed for Python and C++ environments.

## Running the Source code Directly
### Python Version
1. Install dependencies.
2. Start MySQL: `brew services start mysql`
3. Run Script: `python3 -m source_python.dl_inference`
4. Stop MySQL: `brew services stop mysql`

### C++ Version
1. Install dependencies (boost, gsl, eigen)
2. Navigate to source_c++ folder
3. Compile: `g++ -std=c++17 -o executable dl_inference.cpp RateStateModel.cpp MCMC.cpp [...]`

### Features
- Rate State Model Initialization with customizable parameters.
- Time Series Generation and Bayesian Inference using MCMC.
- Visualization with `matplotlib`.

### Customization
Adjust parameters like `number_slip_values`, `lowest_slip_value`, `largest_slip_value`, and `qstart` as needed.

## Running FastAPI
1. Install uvicorn
2. Navigate to fast_api folder
3. Run the main script: `uvicorn main:app --reload`

## Paper Summary
Based on "Arriving at estimates of a rate and state fault friction model parameter using Bayesian inference and Markov chain Monte Carlo
Author links open overlay panel", 
https://www.sciencedirect.com/science/article/pii/S266654412200003X

## Contact
For support or collaboration: dana.spk5@gmail.com

## License
Licensed under [LICENSE NAME](path/to/LICENSE).

---
Copyright © [Saumik Dana]
