import numpy as np
from scipy.stats import gamma 

class MCMC:
    """
    Class for MCMC sampling
    """
    def __init__(
        self, model, data, qpriors, qstart, nsamples=100, lstm_model={}, adapt_interval=10, verbose=True):

        self.model          = model
        self.qstart         = qstart
        self.qpriors        = qpriors
        self.nsamples       = nsamples # number of samples to take during MCMC algorithm
        self.nburn          = int(nsamples/2) # number of samples to discard during burn-in period
        self.verbose        = verbose
        self.adapt_interval = adapt_interval
        self.data           = data
        self.lstm_model     = lstm_model
        self.n0             = 0.01
        self.qstart_limits  = np.array([[self.qpriors[1], self.qpriors[2]]])

        return

    def evaluate_model(self):

        if self.lstm_model:
            acc = self.model.rom_evaluate(self.lstm_model)[1]
        else:
            acc = self.model.evaluate()[1]

        return acc

    def update_standard_deviation(self, SSqprev):

        # Update the estimate of the standard deviation
        aval           = 0.5*(self.n0+len(self.data))
        bval           = 0.5*(self.n0*self.std2[-1]+SSqprev)
        self.std2.append(1/gamma.rvs(aval,scale=1/bval,size=1)[0])

    def update_covariance_matrix(self, qparams):

        # Update the estimate of the covariance matrix
        Vnew = 2.38**2/len(self.qpriors.keys())*np.cov(qparams[:,-self.adapt_interval:])
        if qparams.shape[0]==1:
            Vnew = np.reshape(Vnew,(-1,1))
        Vnew = np.linalg.cholesky(Vnew)
        return Vnew.copy()

    def compute_initial_covariance(self):

        # Evaluate the model with the original dc value
        acc_ = self.evaluate_model()

        # Perturb the dc value
        self.model.Dc *= (1 + 1e-6)

        # Evaluate the model with the perturbed dc value
        acc_dq_ = self.evaluate_model()

        # Extract the values and reshape them to a 1D array
        acc = acc_.reshape(1, -1)
        acc_dq = acc_dq_.reshape(1, -1)

        # Compute the variance of the noise
        self.std2 = [np.sum((acc - self.data) ** 2, axis=1).item()/(acc.shape[1] - len(self.qpriors))]

        # Compute the covariance matrix
        X = ((acc_dq - acc) / (self.model.Dc * 1e-6)).T
        X = np.linalg.inv(np.dot(X.T, X))
        self.Vstart = self.std2[-1] * X 

    def acceptreject(self, q_new, SSqprev, std2):
        """ 
        The acceptreject function checks if a proposed sample falls within the limits 
        and decides whether to accept or reject it 
        based on the acceptance probability and a random number. 
        It returns a boolean value indicating acceptance 
        and the sum of squares error of the accepted or previous proposal.

        The numpy function np.clip() is used to limit the values in an array. 
        Here, it is being used to limit the acceptance probability calculated 
        in the Adaptive Metropolis Algorithm.

        In the provided Python code snippet, np.clip(0.5 * (SSqprev - SSqnew) / std2, -np.inf, 0) 
        is used to ensure that the calculated acceptance probability value does not exceed 0 
        and does not go below negative infinity. 
        This is done by 'clipping' any calculated value below -np.inf to -np.inf, 
        and any calculated value above 0 to 0.

        The expression 0.5 * (SSqprev - SSqnew) / std2 is essentially the log 
        of the acceptance probability used in the Metropolis-Hastings Algorithm. 
        In the usual Metropolis-Hastings Algorithm, 
        we use the ratio of the target densities evaluated at the new and old state. 
        However, when we work with the log of the acceptance probability (as in this case), 
        we subtract these log-densities instead of dividing them.

        SSqprev and SSqnew represent the sum of squares errors for the old and new state, 
        respectively. These are used to calculate the acceptance probability. 
        std2 is a scaling factor.
        """
        # Check if the proposal values are within the limits
        condition1 = (q_new > self.qstart_limits[:, 0])
        condition2 = (q_new < self.qstart_limits[:, 1])
        accept = np.all(condition1 & condition2, axis=0)

        if accept:
            # Compute the sum of squares error of the new proposal
            SSqnew = self.SSqcalc(q_new)

            # Compute the acceptance probability
            accept_prob = np.clip(0.5 * (SSqprev - SSqnew) / std2, -np.inf, 0)

            # Check if the proposal is accepted 
            # based on the acceptance probability and a random number
            accept = accept_prob > np.log(np.random.rand(1))[0]

        if accept:
            # If accepted, return the boolean True and the sum of squares error of the new proposal
            return accept, SSqnew
        else:
            # If rejected, return the boolean False and the sum of squares error of the previous proposal
            return accept, SSqprev

    def SSqcalc(self, q_new):
        """ 
        The SSqcalc function updates the Dc parameter of the model with a proposed value, 
        evaluates the model's performance, and computes the sum of squares error 
        between the model's accuracy and the data.
        """
        # Update the Dc parameter of the model with the new proposal
        self.model.Dc = q_new[0,]

        # Evaluate the model's performance on the problem type and LSTM model
        acc = self.evaluate_model()
        
        # Compute the sum of squares error between the model's accuracy and the data
        SSq = np.sum((acc.reshape(1, -1) - self.data)**2, axis=1, keepdims=True)

        return SSq

    def sample(self):
        """ 
        The code provided seems to be part of a sampling algorithm 
        that uses the Metropolis-Hastings algorithm to generate samples from a distribution. 
        Markov Chain Monte Carlo (MCMC) method for parameter estimation.
        """

        # Compute initial covariance
        self.compute_initial_covariance()
        
        qparams = np.copy(np.array([[self.qstart]])) # Array of sampled parameters
        Vold = np.copy(self.Vstart) # Covariance matrix of previously sampled parameters
        SSqprev = self.SSqcalc(qparams) # Squared error of previously sampled parameters
        iaccept = 0 # Counter for accepted samples

        for isample in np.arange(self.nsamples):
            # Sample new parameters from a normal distribution 
            # with mean being the last element of qparams
            q_new = np.reshape(np.random.multivariate_normal(qparams[:,-1],Vold),(-1,1)) 

            # Accept or reject the new sample based on the Metropolis-Hastings acceptance rule
            accept,SSqnew = self.acceptreject(q_new,SSqprev,self.std2[-1])

            # Print some diagnostic information
            print(isample,accept)
            print("Generated sample ---- ",np.asscalar(q_new))

            if accept:
                # If the new sample is accepted, 
                # add it to the list of sampled parameters
                qparams = np.concatenate((qparams,q_new),axis=1)
                SSqprev = SSqnew.copy()
                iaccept += 1
            else:
                # If the new sample is rejected, 
                # add the previous sample to the list of sampled parameters
                q_new = np.reshape(qparams[:,-1],(-1,1))
                qparams = np.concatenate((qparams,q_new),axis=1)

            self.update_standard_deviation(SSqprev)
            # Update standard deviation

            # Update the covariance matrix if it is time to adapt it
            if (isample+1) % self.adapt_interval == 0:
                try:
                    Vold = self.update_covariance_matrix(qparams)
                except:
                    pass

        # Print acceptance ratio
        print("acceptance ratio:",iaccept/self.nsamples)

       # Trim the estimate of the standard deviation to exclude burn-in samples  
        self.std2 = np.asarray(self.std2)[self.nburn:] 
        
        # Return accepted samples
        return qparams[:,self.nburn:]
