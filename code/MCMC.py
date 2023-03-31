import numpy as np
import copy
from scipy.stats import gamma 
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class MCMC:
    """
    Class for MCMC sampling
    """
    def __init__(self, model, data, qpriors, qstart, nsamples=100, problem_type='full', 
        lstm_model={}, adapt_interval=100, verbose=True):

        self.model             = model
        self.qstart            = qstart
        self.qpriors           = qpriors
        self.nsamples          = nsamples # number of samples to take during MCMC algorithm
        self.nburn             = int(nsamples/2) # number of samples to discard during burn-in period
        self.verbose           = verbose
        self.adapt_interval    = adapt_interval
        self.data              = data
        self.lstm_model        = lstm_model
        self.problem_type      = problem_type
        self.n0                = 0.01

        # Construct the initial covariance matrix
        if self.problem_type == 'full':
            _, acc_, _         = self.model.evaluate()
        else:
            _, acc_            = self.model.rom_evaluate(lstm_model)
        acc                    = acc_[:, 0]
        acc                    = acc.reshape(1, acc.shape[0])
        denom                  = acc.shape[1] 
        denom                 -= len(self.qpriors)
        num                    = np.sum((acc - self.data) ** 2, axis=1)
        self.std2              = [num.item()/denom]

        self.model.Dc         *= (1+1e-6) # perturb the dc
        if self.problem_type == 'full':
            _, acc_dq_, _      = self.model.evaluate()
        else:
            _, acc_dq_         = self.model.rom_evaluate(lstm_model)
        acc_dq                 = acc_dq_[:, 0]
        acc_dq                 = acc_dq.reshape(1, acc_dq.shape[0])
        X                      = []
        X.append((acc_dq[0, :] - acc[0, :])/(self.model.Dc * 1e-6))
        X                      = np.asarray(X).T
        X                      = np.linalg.inv(np.dot(X.T, X))
        self.Vstart            = self.std2[-1] * X  

        # Set up the initial qstart vector and qstart limits
        self.qstart_vect       = np.zeros((1, 1))
        self.qstart_limits     = np.zeros((1, 2))
        self.qstart_vect[0, 0] = self.qstart
        self.qstart_limits[0, 0] = self.qpriors[1]
        self.qstart_limits[0, 1] = self.qpriors[2]
            
        return

    def sample(self):
        """
        Function for sampling using adaptive Metropolis algorithm

        Return:
            Q_MCMC: Accepted samples
        """

        # Initialize variables
        qparams           = copy.deepcopy(self.qstart_vect) # Array of sampled parameters
        Vold              = copy.deepcopy(self.Vstart) # Covariance matrix of previously sampled parameters
        Vnew              = copy.deepcopy(self.Vstart) # Covariance matrix of current sampled parameters
        SSqprev           = self.SSqcalc(qparams) # Squared error of previously sampled parameters
        iaccept           = 0 # Counter for accepted samples

        # Loop over number of desired samples
        for isample in range(0,self.nsamples):

            # Sample new parameters from a normal distribution with mean being the last element of qparams
            q_new         = np.reshape(np.random.multivariate_normal(qparams[:,-1],Vold),(-1,1)) 

            # Accept or reject the new sample based on the Metropolis-Hastings acceptance rule
            accept,SSqnew = self.acceptreject(q_new,SSqprev,self.std2[-1])

            # Print some diagnostic information
            print(isample,accept)
            print("Generated sample ---- ",np.asscalar(q_new))

            # If the new sample is accepted, add it to the list of sampled parameters
            if accept:
                qparams   = np.concatenate((qparams,q_new),axis=1)
                SSqprev   = copy.deepcopy(SSqnew)
                iaccept  += 1
            else:
                # If the new sample is rejected, add the previous sample to the list of sampled parameters
                q_new     = np.reshape(qparams[:,-1],(-1,1))
                qparams   = np.concatenate((qparams,q_new),axis=1)

            # Update the estimate of the standard deviation
            aval          = 0.5*(self.n0+self.data.shape[1])
            bval          = 0.5*(self.n0*self.std2[-1]+SSqprev)
            self.std2.append(1/gamma.rvs(aval,scale=1/bval,size=1)[0])

            # Update the covariance matrix if it is time to adapt it
            if np.mod((isample+1),self.adapt_interval)==0:
                try:
                    Vnew     = 2.38**2/len(self.qpriors.keys())*np.cov(qparams[:,-self.adapt_interval:])
                    if qparams.shape[0]==1:
                        Vnew = np.reshape(Vnew,(-1,1))
                    R = np.linalg.cholesky(Vnew)
                    Vold = copy.deepcopy(Vnew)
                except:
                    pass

        # Print acceptance ratio
        print("acceptance ratio:",iaccept/self.nsamples)

        # Return accepted samples
        self.std2 = np.asarray(self.std2)
        self.std2 = self.std2[self.nburn:] # Trim the estimate of the standard deviation to exclude burn-in samples
        return qparams[:,self.nburn:]

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

    def SSqcalc(self, q_new):
        """
        Compute the sum of squares error of the proposed parameter values.

        Parameters:
            q_new (numpy array): A numpy array representing the proposed values for the parameters.

        Returns:
            numpy array: A numpy array representing the sum of squares error of the proposed parameter values.
        """
        self.model.Dc  = q_new[0,]
        if self.problem_type == 'full': 
            _, acc_, _ = self.model.evaluate() # high fidelity model
        else: 
            _, acc_    = self.model.rom_evaluate(self.lstm_model) # reduced order model
        acc            = acc_[:, 0] 
        acc            = acc.reshape(1, acc.shape[0])
        SSq            = np.sum((acc - self.data)**2, axis=1) # squared error

        return SSq

    def plot_dist(self, qparams, dc):
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

        return 