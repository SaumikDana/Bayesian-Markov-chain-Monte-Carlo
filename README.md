## Deep learning and Bayesian inference

This code is a Python implementation of the adaptive Metropolis algorithm for Markov Chain Monte Carlo (MCMC) sampling. The purpose of this code is to sample the posterior distribution of a model's parameters given some observed data. The code is structured as a class called "MCMC" with four methods:

    init: Initializes the sampling process with the necessary parameters and constructs the initial covariance matrix.
    sample: Performs the MCMC sampling using the adaptive Metropolis algorithm, updating the covariance matrix at specified intervals.
    acceptreject: Determines whether a new sample should be accepted or rejected based on the calculated squared error and random chance.
    SSqcalc: Calculates the squared error for a given set of parameters.
    plot_dist: Plots the distribution of the samples and saves the plot to a file.

The code requires the following Python packages: numpy, copy, scipy, sys, and matplotlib.

To use the code, instantiate an object of the MCMC class and call the sample() method. This will perform the MCMC sampling and return the accepted samples. The plot_dist() method can be used to plot the distribution of the samples and save the plot to a file.

This code is meant to be used as a tool for Bayesian inference, where a model's parameters are estimated by sampling from the posterior distribution using MCMC.

