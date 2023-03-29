from rsf import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':

    #Usage: python dl_inference.py -epochs 100 -samples 100 --bayesian
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs',dest='num_epochs', type=int, help="Number of Epochs")
    parser.add_argument('-samples',dest='num_samples', type=int, help="Number of Samples")
    parser.add_argument('--reduction', dest='reduction', action='store_true')
    parser.add_argument('--overlap', dest='overlap', action='store_true')
    parser.add_argument('--bayesian', dest='bayesian', action='store_true')

    args = parser.parse_args()

    # rsf problem
    problem = rsf(args)
    
    # Solve the problem
    problem.model.plotfigs = True
    problem.solve()

    # Close it out
    plt.show()
    plt.close('all')

