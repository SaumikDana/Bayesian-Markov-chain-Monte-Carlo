# Bayesian MCMC in Rate State Models

## Introduction
`dl_inference` is a tool that implements a Rate State model and performs Bayesian inference using MCMC sampling, designed for Python and C++ environments.

## Installation

### Python Version
1. Install dependencies.
2. Start MySQL: `brew services start mysql`
3. Run Script: `python dl_inference.py`
4. Stop MySQL: `brew services stop mysql`

### C++ Version
Install dependencies (boost, gsl, eigen), then compile:
`g++ -std=c++17 -o executable dl_inference.cpp RateStateModel.cpp MCMC.cpp [...]`


## Features
- Rate State Model Initialization with customizable parameters.
- Time Series Generation and Bayesian Inference using MCMC.
- Visualization with `matplotlib`.

## Customization
Adjust parameters like `number_slip_values`, `lowest_slip_value`, `largest_slip_value`, and `qstart` as needed.

## Paper Summary
Based on "Bayesian Inference Framework for Rate and State Fault Friction Models", this tool estimates the critical slip distance in fault friction models.

## How to Contribute
See [CONTRIBUTING.md](path/to/CONTRIBUTING.md) for guidelines.

## Troubleshooting
Visit our [FAQ section](path/to/FAQ.md) for common issues and solutions.

## Contact
For support or collaboration: dana.spk5@gmail.com

## License
Licensed under [LICENSE NAME](path/to/LICENSE).

---
Copyright © [Saumik Dana]
