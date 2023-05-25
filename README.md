# Python version

dl_inference is a Python script that implements a Rate State model and performs Bayesian inference on the generated time series. It utilizes the `rsf` module and `matplotlib` library. A version that uses deep learning to generate a reduced order Rate State model is under development.

## Prerequisites
- Python 3.x
- `rsf` module
- `matplotlib` library

## Usage
1. Install the required dependencies.
2. Run the script using a Python interpreter.
    ```shell
    python dl_inference.py
    ```

## Description
The code performs the following steps:

1. Initializes the Rate State model with the specified parameters: `number_slip_values`, `lowest_slip_value`, `largest_slip_value`, and `plotfigs` (for plotting figures).
2. Generates the time series for the Rate State model.
3. Sets the starting value for the parameter `qstart`.
4. Defines the prior distribution for the parameter `qstart` using the "Uniform" distribution with bounds [0, 10000].
5. Performs Bayesian inference by calling the `inference` method with `nsamples=500`.
6. Displays the generated figures using `matplotlib`.
7. Closes all open figures.

## Customization
- You can modify the parameters `number_slip_values`, `lowest_slip_value`, `largest_slip_value`, and `plotfigs` to fit your specific needs.
- Adjust the `qstart` parameter and its prior distribution to tailor the Bayesian inference.
- Modify the `nsamples` parameter in the `inference` method to control the number of samples used in the inference process.

# C++ version

```
g++ -std=c++17 -o executable dl_inference.cpp fileIO.cpp -I/opt/homebrew/Cellar/boost/1.81.0_1/include -I/opt/homebrew/Cellar/gsl/2.7.1/include -L/opt/homebrew/Cellar/boost/1.81.0_1/lib -lboost_iostreams -L/opt/homebrew/Cellar/gsl/2.7.1/lib -lgsl -lgslcblas -lm

```

## Contact
dana.spk5@gmail.com