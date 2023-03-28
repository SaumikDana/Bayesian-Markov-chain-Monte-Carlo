from matplotlib.pyplot import flag
import numpy as np
import copy
from scipy.stats import gamma 
import sys
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

"""

__init__: Initializes the sampling process with the necessary parameters and constructs the initial covariance matrix.

sample: Performs the MCMC sampling using the adaptive Metropolis algorithm, updating the covariance matrix at specified intervals.

acceptreject: Determines whether a new sample should be accepted or rejected based on the calculated squared error and random chance.

SSqcalc: Calculates the squared error for a given set of parameters.

plot_dist: Plots the distribution of the samples and saves the plot to a file.

"""

class MCMC:
    """
    Class for MCMC sampling
    """
    def __init__(self, model, qpriors, nsamples, nburn, data, problem_type, lstm_model, qstart=None, adapt_interval=100, verbose=True):
        """
        Initialize the sampling process
        """
        self.model = model
        self.qstart = qstart
        self.qpriors = qpriors
        self.nsamples = nsamples
        self.nburn = nburn
        self.verbose = verbose
        self.adapt_interval = adapt_interval
        self.data = data
        self.lstm_model = lstm_model
        self.problem_type = problem_type
        self.consts = model.consts
        self.n0 = 0.01

        # Set the initial guess using the qstart dictionary
        for arg in self.qstart.keys():
            self.consts[arg] = qstart[arg]

        # Evaluate the model based on the problem type (full or reduced order model)
        if self.problem_type == 'full':
            # High fidelity model
            t_, acc_, temp_ = self.model.evaluate(self.consts)
        else:
            # Reduced order model
            t_, acc_ = self.model.rom_evaluate(self.consts, lstm_model)

        # Construct the initial covariance matrix
        acc = acc_[:, 0]
        acc = acc.reshape(1, acc.shape[0])
        self.std2 = [np.sum((acc - self.data) ** 2, axis=1)[0] / (acc.shape[1] - len(self.qpriors.keys()))]
        X = []

        # Iterate through the qpriors keys to construct the initial covariance matrix
        for arg in self.qpriors.keys():
            consts_dq = copy.deepcopy(self.consts)
            consts_dq[arg] = consts_dq[arg] * (1 + 1e-6)

            # Evaluate the model based on the problem type (full or reduced order model)
            if self.problem_type == 'full':  # High fidelity model
                t_, acc_dq_, temp_ = self.model.evaluate(consts_dq)
            else:  # Reduced order model
                t_, acc_dq_ = self.model.rom_evaluate(consts_dq, lstm_model)

            acc_dq = acc_dq_[:, 0]
            acc_dq = acc_dq.reshape(1, acc_dq.shape[0])
            X.append((acc_dq[0, :] - acc[0, :]) / (consts_dq[arg] * 1e-6))

        # Compute the inverse of the product of the transpose of X and X
        X = np.asarray(X).T
        X = np.linalg.inv(np.dot(X.T, X))
        self.Vstart = self.std2[0] * X  # Construct the initial covariance matrix

        # Set up the initial qstart vector and qstart limits
        self.qstart_vect = np.zeros((len(self.qstart), 1))
        self.qstart_limits = np.zeros((len(self.qstart), 2))
        flag = 0
        for arg in self.qstart.keys():
            self.qstart_vect[flag, 0] = self.qstart[arg]
            self.qstart_limits[flag, 0] = self.qpriors[arg][1]
            self.qstart_limits[flag, 1] = self.qpriors[arg][2]
            flag = flag + 1

    def sample(self):
        """
        Function for sampling using adaptive Metropolis algorithm

        Return:
            Q_MCMC: Accepted samples
        """

        # Initialize variables
        qparams=copy.deepcopy(self.qstart_vect) # Array of sampled parameters
        qmean_old=copy.deepcopy(self.qstart_vect) # Mean of previously sampled parameters
        qmean=copy.deepcopy(self.qstart_vect) # Mean of current sampled parameters
        Vold=copy.deepcopy(self.Vstart) # Covariance matrix of previously sampled parameters
        Vnew=copy.deepcopy(self.Vstart) # Covariance matrix of current sampled parameters
        SSqprev=self.SSqcalc(qparams) # Squared error of previously sampled parameters
        iaccept=0 # Counter for accepted samples

        # Loop over number of desired samples
        for isample in range(0,self.nsamples):

            # Sample new parameters from a normal distribution with mean being the last element of qparams
            q_new = np.reshape(np.random.multivariate_normal(qparams[:,-1],Vold),(-1,1)) 

            # Accept or reject the new sample based on the Metropolis-Hastings acceptance rule
            accept,SSqnew=self.acceptreject(q_new,SSqprev,self.std2[-1])

            # Print some diagnostic information
            print(isample,accept)
            print("Generated sample ---- ",np.asscalar(q_new))

            # If the new sample is accepted, add it to the list of sampled parameters
            if accept:
                qparams=np.concatenate((qparams,q_new),axis=1)
                SSqprev=copy.deepcopy(SSqnew)
                iaccept=iaccept+1
            else:
                # If the new sample is rejected, add the previous sample to the list of sampled parameters
                q_new=np.reshape(qparams[:,-1],(-1,1))
                qparams=np.concatenate((qparams,q_new),axis=1)

            # Update the estimate of the standard deviation
            aval=0.5*(self.n0+self.data.shape[1]);
            bval=0.5*(self.n0*self.std2[-1]+SSqprev);
            self.std2.append(1/gamma.rvs(aval,scale=1/bval,size=1)[0])

            # Update the covariance matrix if it is time to adapt it
            if np.mod((isample+1),self.adapt_interval)==0:
                try:
                    Vnew=2.38**2/len(self.qpriors.keys())*np.cov(qparams[:,-self.adapt_interval:])
                    if qparams.shape[0]==1:
                        Vnew=np.reshape(Vnew,(-1,1))
                    R = np.linalg.cholesky(Vnew)
                    Vold=copy.deepcopy(Vnew)
                except:
                    pass

        # Print acceptance ratio
        print("acceptance ratio:",iaccept/self.nsamples)

        # Return accepted samples
        self.std2=np.asarray(self.std2)[self.nburn:] # Trim the estimate of the standard deviation to exclude burn-in samples
        return qparams[:,self.nburn:]

    """
    The acceptreject() function is a component of the Metropolis-Hastings algorithm, 
    and implements the accept-reject step. Given a proposed new set of parameter values q_new, 
    the function computes whether or not to accept these values as the new state of the Markov chain. 
    If the new values are within the specified limits, 
    the function computes the sum of squares error of the proposed state SSqnew, 
    as well as the acceptance probability. 
    If the acceptance probability is greater than a random number drawn from a uniform distribution between 0 and 1, 
    then the proposal is accepted. 
    The function returns a tuple containing a boolean indicating whether the proposal is accepted or rejected, 
    and the sum of squares error of the proposal (either the previous or the new one).
    """
    def acceptreject(self, q_new, SSqprev, std2):
        """
        Implementation of the accept-reject step in Metropolis-Hastings algorithm.

        Parameters:
            q_new (numpy array): A numpy array representing the new proposal values for the parameters.
            SSqprev (float):     A float representing the sum of squares error of the previous proposal.
            std2 (float):        A float representing the variance of the distribution used to generate the proposal.

        Returns:
            tuple: A tuple containing a boolean indicating whether the proposal is accepted or rejected, and 
                   the sum of squares error of the proposal (either the previous or the new one).
        """
        # Check if the proposal values are within the limits
        accept = np.all(q_new[:, 0] > self.qstart_limits[:, 0]) and np.all(q_new[:, 0] < self.qstart_limits[:, 1])

        if accept:
            # Compute the sum of squares error of the new proposal
            SSqnew = self.SSqcalc(q_new)
            # Compute the acceptance probability
            accept_prob = min(0, 0.5*(SSqprev - SSqnew) / std2)
            # Check if the proposal is accepted based on the acceptance probability and a random number
            accept = accept_prob > np.log(np.random.rand(1))[0]

        if accept:
            # If accepted, return the boolean True and the sum of squares error of the new proposal
            return accept, SSqnew
        else:
            # If rejected, return the boolean False and the sum of squares error of the previous proposal
            return accept, SSqprev

    """
    The SSqcalc() function is a helper function that computes the sum of squares error of the proposed parameter values. 
    The function takes the proposed parameter values as input, 
    and returns a numpy array representing the sum of squares error. 
    The function first copies the original self.consts dictionary to consts_dq and updates 
    the values of the parameters to the proposed values. Then, depending on the problem type, 
    the function either calls the evaluate() method of the high fidelity model or the rom_evaluate() 
    method of the reduced order model to compute the response. 
    The computed response is then reshaped to a 1-dimensional numpy array and the squared error 
    is computed as the sum of squares of the difference between the computed response and the target data. 
    The function returns the squared error as a numpy array.
    """            
    def SSqcalc(self, q_new):
        """
        Compute the sum of squares error of the proposed parameter values.

        Parameters:
            q_new (numpy array): A numpy array representing the proposed values for the parameters.

        Returns:
            numpy array: A numpy array representing the sum of squares error of the proposed parameter values.
        """
        flag = 0
        for arg in self.qstart.keys(): 
            consts_dq = copy.deepcopy(self.consts)
            consts_dq[arg] = q_new[flag, ]
            flag += 1

        if self.problem_type == 'full': 
            # high fidelity model
            t_, acc_, temp_ = self.model.evaluate(consts_dq)
        else: 
            # reduced order model
            t_, acc_ = self.model.rom_evaluate(consts_dq, self.lstm_model)

        acc = acc_[:, 0] 
        acc = acc.reshape(1, acc.shape[0])
        SSq = np.sum((acc - self.data)**2, axis=1) # squared error

        return SSq


    def plot_dist(self, qparams, plot_title, dc):
        # Set up the plot layout with 1 row and 2 columns, and adjust the width ratios and space between subplots
        n_rows = 1
        n_columns = 2
        gridspec = {'width_ratios': [0.7, 0.15], 'wspace': 0.15}

        # Create the subplots with the specified gridspec
        fig, ax = plt.subplots(n_rows, n_columns, gridspec_kw=gridspec)

        # Set the main plot title with the specified dc value
        fig.suptitle(f'$d_c={dc}\,\mu m$', fontsize=14)

        # Plot the MCMC samples as a blue line in the first subplot
        ax[0].plot(qparams[0, :], 'b-', linewidth=1.0)

        # Get the limits of the y-axis
        ylims = ax[0].get_ylim()

        # Create an array of 1000 evenly spaced points between the y-axis limits
        x = np.linspace(ylims[0], ylims[1], 1000)

        # Calculate the probability density function using Gaussian Kernel Density Estimation
        kde = gaussian_kde(qparams[0, :])

        # Plot the probability density function as a blue line in the second subplot
        ax[1].plot(kde.pdf(x), x, 'b-')

        # Find the maximum value of the probability density function
        max_val = x[kde.pdf(x).argmax()]

        # Plot the maximum value as a red dot and annotate it with the value
        ax[1].plot(kde.pdf(x)[kde.pdf(x).argmax()], max_val, 'ro')
        ax[1].annotate(str(round(max_val, 2)), xy=(0.90 * kde.pdf(x)[kde.pdf(x).argmax()], max_val), size=14)

        # Fill the area under the probability density function with a light blue color
        ax[1].fill_betweenx(x, kde.pdf(x), np.zeros(x.shape), alpha=0.3)

        # Set the x-axis limits for the second subplot
        ax[1].set_xlim(0, None)

        # Set labels and axis limits for the first subplot
        ax[0].set_ylabel('$d_c$', fontsize=14)
        ax[0].set_xlim(0, qparams.shape[1])
        ax[0].set_xlabel('Sample number')

        # Set the x-axis label for the second subplot and hide the y-axis
        ax[1].set_xlabel('Prob. density')
        ax[1].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(True)
        ax[1].get_xaxis().set_ticks([])

        # Create an output directory for the plots if it doesn't already exist
        output_path = Path('plots')
        output_path.mkdir(parents=True, exist_ok=True)

        # Save the figure to the output directory with the specified file name
        fig.savefig(output_path / f'burn_{plot_title}_{dc}_{sys.argv[2]}_.png')
