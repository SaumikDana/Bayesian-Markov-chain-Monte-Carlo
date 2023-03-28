# Deep learning and Bayesian inference

## dl_inference.py

This is a Python script for performing deep learning inference for a geophysical problem of type RSF. The script uses command line arguments to specify the number of epochs, number of samples, and other options such as reduction, overlap, and Bayesian inference.
Dependencies

    rsf
    numpy
    matplotlib
    argparse

### Installation

    Clone the repository.
    Install the required packages by running: pip install -r requirements.txt

### Usage

To use this script, navigate to the directory containing the script and run the following command:

    python dl_inference.py -epochs [num_epochs] -samples [num_samples] [--reduction] [--overlap] [--bayesian]

    num_epochs: Specify the number of epochs for training the neural network.
    num_samples: Specify the number of samples to use for inference.
    --reduction: Optional argument to use data reduction technique for solving the problem.
    --overlap: Optional argument to use overlap technique for solving the problem.
    --bayesian: Optional argument to use Bayesian inference for solving the problem.

### Example

To solve an RSF problem with 100 epochs and 100 samples, with data reduction and overlap techniques applied, run the following command:

    python dl_inference.py -epochs 100 -samples 100 --reduction --overlap
### Output

The script generates visualizations of the solved problem using the matplotlib.pyplot module. The visualizations are displayed using plt.show() and closed using plt.close('all').

## MCMC.py

This code is a Python implementation of the adaptive Metropolis algorithm for Markov Chain Monte Carlo (MCMC) sampling. The purpose of this code is to sample the posterior distribution of a model's parameters given some observed data. The code is structured as a class called "MCMC" with four methods:

    init: Initializes the sampling process with the necessary parameters and constructs the initial covariance matrix.
    sample: Performs the MCMC sampling using the adaptive Metropolis algorithm, updating the covariance matrix at specified intervals.
    acceptreject: Determines whether a new sample should be accepted or rejected based on the calculated squared error and random chance.
    SSqcalc: Calculates the squared error for a given set of parameters.
    plot_dist: Plots the distribution of the samples and saves the plot to a file.

The code requires the following Python packages: numpy, copy, scipy, sys, and matplotlib.

To use the code, instantiate an object of the MCMC class and call the sample() method. This will perform the MCMC sampling and return the accepted samples. The plot_dist() method can be used to plot the distribution of the samples and save the plot to a file.

This code is meant to be used as a tool for Bayesian inference, where a model's parameters are estimated by sampling from the posterior distribution using MCMC.

