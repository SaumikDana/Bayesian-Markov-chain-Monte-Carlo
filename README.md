# Bayesian MCMC in Rate State Models

## Directory Structure
```
Bayesian_MCMC_Deep-Learning/
│
├── .github/
│   └── workflows/
│       └── python-app.yml
│
├── .vscode/
│   ├── launch.json
│   ├── settings.json
│   └── tasks.json
│
├── source_c++/
│   ├── MCMC.cpp
│   ├── MCMC.h
│   ├── RateStateModel.cpp
│   ├── RateStateModel.h
│   └── dl_inference.cpp
│
├── source_python/
│   ├── MCMC.py
│   ├── RSF.py
│   ├── RateStateModel.py
│   ├── __init__.py
│   ├── dl_inference.py
│   ├── lstm/
│   │   ├── lstm_encoder_decoder.py
│   │   └── utils.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_mcmc.py
│   │   └── test_rsf.py
│   └── utils/
│       ├── json_save_load.py
│       └── mysql_save_load.py
│
├── .DS_Store
├── README.md
├── __init__.py
└── requirements.txt
```

## Introduction
`dl_inference` is a tool that implements a Rate State model and performs Bayesian inference using MCMC sampling, designed for Python and C++ environments.

## Running the Source code Directly
### Python Version
1. Install dependencies.
2. Run Script: `python3 -m source_python.dl_inference`

### C++ Version
1. Install dependencies (boost, gsl, eigen)
2. Navigate to source_c++ folder
3. Compile: `g++ -std=c++17 -o executable dl_inference.cpp RateStateModel.cpp MCMC.cpp [...]`
4. Run executable

### Features
- Rate State Model Initialization with customizable parameters.
- Time Series Generation and Bayesian Inference using MCMC.
- Visualization with `matplotlib`.

## Paper Summary
Based on "Arriving at estimates of a rate and state fault friction model parameter using Bayesian inference and Markov chain Monte Carlo", https://www.sciencedirect.com/science/article/pii/S266654412200003X

## Contact
For support or collaboration: dana.spk5@gmail.com

---
Copyright © [Saumik Dana]
---
