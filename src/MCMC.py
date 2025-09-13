import setup_path
from src.imports import *


class MCMC:
    """
    Adaptive Metropolis-Hastings Markov Chain Monte Carlo (MCMC) sampler.
    
    This class implements an adaptive MCMC algorithm for Bayesian parameter estimation.
    It uses the Metropolis-Hastings algorithm with adaptive covariance matrix updates
    to improve sampling efficiency. The algorithm is particularly suited for sampling
    from posterior distributions in inverse problems where direct sampling is difficult.
    
    The adaptive feature automatically adjusts the proposal covariance matrix during
    sampling to achieve better acceptance rates and mixing. The algorithm also
    incorporates hierarchical Bayesian modeling by updating the noise variance
    (standard deviation) at each step.
    
    Attributes:
        model: The forward model used for evaluation
        qstart (float): Initial parameter value for sampling
        qpriors (dict): Prior parameter bounds and information
        nsamples (int): Total number of MCMC samples to generate
        nburn (int): Number of burn-in samples to discard
        verbose (bool): Whether to print diagnostic information
        adapt_interval (int): Frequency of covariance matrix adaptation
        data (np.ndarray): Observed data for comparison
        lstm_model (dict): Optional LSTM model configuration
        n0 (float): Prior shape parameter for noise variance
        qstart_limits (np.ndarray): Parameter bounds for acceptance
        dc_true (float): True parameter value (for comparison/validation)
        std2 (list): Evolution of noise variance estimates
        Vstart (np.ndarray): Initial covariance matrix
    
    Example:
        >>> # Setup model and data
        >>> model = MyForwardModel()
        >>> data = np.array([1.2, 1.5, 1.8])
        >>> priors = {1: 0.5, 2: 2.0}  # lower_bound, upper_bound
        >>> 
        >>> # Initialize MCMC sampler
        >>> mcmc = MCMC(model=model, data=data, dc_true=1.0, 
        ...             qpriors=priors, qstart=1.2, nsamples=1000)
        >>> 
        >>> # Run sampling
        >>> samples = mcmc.sample(MAKE_ANIMATIONS=False)
        >>> print(f"Posterior mean: {np.mean(samples):.3f}")
    """
    
    def __init__(
        self, 
        model, 
        data,
        dc_true, 
        qpriors, 
        qstart, 
        nsamples=100, 
        lstm_model={}, 
        adapt_interval=10, 
        verbose=True
    ):
        """
        Initialize the MCMC sampler with model, data, and sampling parameters.
        
        Args:
            model: Forward model object that can evaluate predictions given parameters.
                  Must have a 'Dc' attribute that can be set and an 'evaluate()' method.
            data (np.ndarray): Observed data vector for likelihood computation.
                             Should be 1D array of observations.
            dc_true (float): True parameter value used for validation and animation labels.
            qpriors (dict): Dictionary containing prior bounds. Expected format:
                          {1: lower_bound, 2: upper_bound} for uniform prior.
            qstart (float): Initial parameter value to start the MCMC chain.
                          Should be within the prior bounds.
            nsamples (int, optional): Total number of MCMC samples to generate. 
                                    Defaults to 100.
            lstm_model (dict, optional): Configuration for LSTM-based reduced order model.
                                       If provided, uses ROM evaluation. Defaults to {}.
            adapt_interval (int, optional): Number of samples between covariance matrix
                                          adaptations. Defaults to 10.
            verbose (bool, optional): Whether to print diagnostic information during
                                    sampling. Defaults to True.
        
        Note:
            - Burn-in period is automatically set to half of nsamples
            - The noise variance prior uses n0=0.01 as the shape parameter
            - Parameter bounds are derived from qpriors for proposal acceptance
        """
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
        self.dc_true        = dc_true

        return

    def evaluate_model(self):
        """
        Evaluate the forward model for current parameter values.
        
        This method provides a unified interface for model evaluation, automatically
        selecting between full model evaluation and reduced-order model (ROM) evaluation
        based on whether an LSTM model is provided.
        
        Returns:
            np.ndarray: Model predictions/outputs for the current parameter values.
                       The specific format depends on the model implementation.
        
        Note:
            - If lstm_model is provided and non-empty, uses ROM evaluation
            - Otherwise uses standard model evaluation
            - The model's 'Dc' attribute should be set before calling this method
        
        Example:
            >>> self.model.Dc = 1.5
            >>> predictions = self.evaluate_model()
        """
        if self.lstm_model:
            return self.model.reduced_order_model.evaluate(self.lstm_model)[1]
        else:
            return self.model.evaluate()[1]
            
    def update_standard_deviation(self, SSqprev):
        """
        Update the noise variance estimate using Bayesian inference.
        
        This method implements a hierarchical Bayesian approach where the noise
        variance (standard deviation squared) is treated as an unknown parameter
        with an inverse-gamma prior. The posterior is also inverse-gamma distributed,
        allowing for analytical updates.
        
        The update follows the inverse-gamma conjugate prior framework:
        - Prior: σ² ~ InverseGamma(n0/2, n0*σ0²/2)
        - Posterior: σ² ~ InverseGamma((n0+n)/2, (n0*σ0² + SSE)/2)
        
        Args:
            SSqprev (float): Sum of squared errors for the current parameter values.
                           Used to update the scale parameter of the inverse-gamma
                           distribution.
        
        Note:
            - Updates self.std2 list with new variance sample
            - Uses the previous variance estimate (self.std2[-1]) in the update
            - The variance sample is drawn from the posterior distribution
            - Accept/reject criterion depends on this updated standard deviation
        
        Mathematical Details:
            - Shape parameter: a = (n0 + len(data)) / 2
            - Scale parameter: b = (n0 * previous_variance + SSE) / 2  
            - New variance: σ² ~ InverseGamma(a, b)
        """
        aval = 0.5*(self.n0+len(self.data))
        bval = 0.5*(self.n0*self.std2[-1]+SSqprev)
        self.std2.append(1/gamma.rvs(aval,scale=1/bval,size=1)[0])

    def update_covariance_matrix(self, qparams):
        """
        Update the proposal covariance matrix for adaptive MCMC.
        
        This method implements the adaptive Metropolis algorithm by updating the
        proposal covariance matrix based on the sample covariance of recent parameter
        samples. This adaptation improves the efficiency of the MCMC sampler by
        learning the appropriate scale and orientation for proposals.
        
        The method follows the optimal scaling theory for MCMC, using:
        - Scaling factor: (2.38)²/d where d is the dimension
        - Sample covariance from the last 'adapt_interval' samples
        - Cholesky decomposition for numerical stability in proposal generation
        
        Args:
            qparams (np.ndarray): Array of parameter samples with shape (n_params, n_samples).
                                The method uses only the last 'adapt_interval' samples
                                for covariance estimation.
        
        Returns:
            np.ndarray: Lower triangular Cholesky factor of the updated covariance matrix.
                       Used for generating correlated proposals via matrix multiplication.
        
        Raises:
            LinAlgError: If the sample covariance matrix is not positive definite.
                        This can happen with insufficient samples or highly correlated chains.
        
        Note:
            - The 2.38² scaling ensures optimal acceptance rate (~23.4%) for Gaussian targets
            - For 1D problems, the covariance is reshaped to maintain matrix structure
            - The Cholesky factor is returned for efficient proposal generation
            - Adaptation should only occur after sufficient samples for stable covariance estimation
        
        Mathematical Details:
            - Optimal covariance: C_opt = (2.38²/d) * Σ_sample
            - Where Σ_sample is the empirical covariance of recent samples
            - Cholesky decomposition: C_opt = L * L^T
        """
        Vnew = 2.38**2/len(self.qpriors.keys())*np.cov(qparams[:,-self.adapt_interval:])
        if qparams.shape[0]==1:
            Vnew = np.reshape(Vnew,(-1,1))
        Vnew = np.linalg.cholesky(Vnew)
        return Vnew.copy()

    def compute_initial_covariance(self):
        """
        Compute the initial proposal covariance matrix using finite differences.
        
        This method estimates the initial covariance matrix for MCMC proposals by
        computing a finite difference approximation of the Hessian matrix. The approach
        perturbs the initial parameter guess and evaluates how the model response changes,
        providing information about the local curvature of the posterior distribution.
        
        The method performs the following steps:
        1. Evaluates the model at the initial parameter guess
        2. Perturbs the parameter by a small amount (1e-6 relative)
        3. Re-evaluates the model with the perturbed parameter
        4. Computes finite difference approximation of the gradient
        5. Estimates the Hessian and its inverse for the covariance matrix
        6. Scales by the noise variance estimate
        
        Updates:
            self.std2 (list): Initializes with the noise variance estimate from residuals
            self.Vstart (np.ndarray): Initial covariance matrix for proposals
        
        Mathematical Details:
            - Finite difference gradient: g ≈ (f(x+δ) - f(x)) / δ
            - Hessian approximation: H ≈ g^T * g (Gauss-Newton approximation)
            - Initial covariance: V₀ = σ² * H⁻¹
            - Noise variance: σ² = SSE / (n - p) where n=data points, p=parameters
        
        Note:
            - Uses a relative perturbation of 1e-6 for finite differences
            - The noise variance is estimated from the initial residuals
            - This provides a reasonable starting point for the adaptive algorithm
            - The method assumes the posterior is approximately Gaussian locally
        
        Example:
            >>> mcmc.compute_initial_covariance()
            >>> print(f"Initial noise variance: {mcmc.std2[0]:.6f}")
            >>> print(f"Initial covariance shape: {mcmc.Vstart.shape}")
        """
        # Initial Guess
        self.model.Dc = self.qstart

        # Evaluate the model on the initial guess                 
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
        Implement the Metropolis-Hastings acceptance/rejection criterion.
        
        This method determines whether to accept or reject a proposed parameter sample
        based on the Metropolis-Hastings algorithm. It first checks if the proposal
        falls within the prior bounds, then computes the acceptance probability based
        on the likelihood ratio, and finally makes a random decision.
        
        The acceptance probability is computed in log-space for numerical stability:
        log(α) = min(0, log(p(y|θ_new)/p(y|θ_old))) 
               = min(0, -0.5 * (SSE_new - SSE_old) / σ²)
        
        Args:
            q_new (np.ndarray): Proposed parameter values with shape (n_params, 1).
            SSqprev (float): Sum of squared errors for the current/previous parameter values.
            std2 (float): Current estimate of the noise variance.
        
        Returns:
            tuple: A tuple containing:
                - accept (bool): True if the proposal is accepted, False otherwise
                - SSq (float): Sum of squared errors for the accepted state
                              (SSqnew if accepted, SSqprev if rejected)
        
        Algorithm Steps:
            1. Check if proposal is within prior bounds (uniform prior)
            2. If within bounds, evaluate model and compute likelihood ratio
            3. Compute log acceptance probability (clipped to [-∞, 0])
            4. Accept if log(α) > log(U) where U ~ Uniform(0,1)
            5. Return acceptance decision and appropriate SSE value
        
        Mathematical Details:
            - Prior bounds check: lower_bound ≤ θ_new ≤ upper_bound
            - Log-likelihood ratio: Δ log L = -0.5 * (SSE_new - SSE_old) / σ²
            - Acceptance probability: α = min(1, exp(Δ log L))
            - Random acceptance: Accept if α > U where U ~ Uniform(0,1)
        
        Note:
            - Uses np.clip to ensure log probability doesn't exceed 0
            - Proposals outside prior bounds are automatically rejected
            - The method handles both parameter bounds and likelihood evaluation
            - Random number generation uses np.random.rand() for uniform samples
        
        Example:
            >>> q_proposal = np.array([[1.5]])
            >>> accepted, sse = self.acceptreject(q_proposal, sse_current, variance_current)
            >>> if accepted:
            ...     print(f"Accepted proposal with SSE: {sse:.4f}")
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

        return accept, SSqnew if accept else SSqprev

    def SSqcalc(self, q_new):
        """
        Calculate the sum of squared errors for proposed parameter values.
        
        This method evaluates the forward model with new parameter values and
        computes the sum of squared errors between model predictions and observed data.
        This quantity is used in the likelihood calculation for the Metropolis-Hastings
        acceptance criterion.
        
        The sum of squared errors (SSE) is a key component of the Gaussian likelihood:
        L(θ) ∝ exp(-SSE/(2σ²))
        where SSE = Σ(y_i - f(x_i, θ))²
        
        Args:
            q_new (np.ndarray): Proposed parameter values with shape (n_params,).
                              For this implementation, expects a 1D array where
                              the first element is the Dc parameter.
        
        Returns:
            np.ndarray: Sum of squared errors with shape (1, 1).
                       Returns as 2D array for consistency with matrix operations.
        
        Process:
            1. Update the model's Dc parameter with the proposed value
            2. Evaluate the forward model to get predictions
            3. Reshape predictions to match data format
            4. Compute squared differences between predictions and data
            5. Sum the squared errors and return as 2D array
        
        Mathematical Details:
            - SSE = Σᵢ (yᵢ - f(xᵢ, θ))²
            - Where yᵢ are observations, f(xᵢ, θ) are model predictions
            - Used in Gaussian log-likelihood: log L = -n/2 log(2πσ²) - SSE/(2σ²)
        
        Note:
            - Modifies self.model.Dc as a side effect
            - Assumes model.evaluate() returns predictions in appropriate format
            - Reshaping handles potential dimension mismatches between predictions and data
            - The keepdims parameter ensures output maintains 2D structure
        
        Example:
            >>> q_test = np.array([1.2])
            >>> sse = self.SSqcalc(q_test)
            >>> print(f"Sum of squared errors: {sse[0,0]:.4f}")
        """
        # Update the Dc parameter of the model with the new proposal
        self.model.Dc = q_new[0,]

        # Evaluate the model's performance on the problem type and LSTM model
        acc = self.evaluate_model()
        
        # Compute the sum of squares error between the model's accuracy and the data
        SSq = np.sum((acc.reshape(1, -1) - self.data)**2, axis=1, keepdims=True)

        return SSq

    def sample(self, MAKE_ANIMATIONS):
        """
        Execute the main MCMC sampling algorithm with adaptive covariance updates.
        
        This method implements the complete adaptive Metropolis-Hastings algorithm,
        including initialization, proposal generation, acceptance/rejection decisions,
        covariance matrix adaptation, and noise variance updates. Optionally creates
        real-time animations of the sampling process.
        
        The algorithm follows these key steps:
        1. Initialize covariance matrix using finite differences
        2. For each iteration:
           - Generate proposal from multivariate normal distribution
           - Apply Metropolis-Hastings acceptance criterion
           - Update noise variance using Bayesian inference
           - Periodically adapt the proposal covariance matrix
        3. Return samples after burn-in period
        
        Args:
            MAKE_ANIMATIONS (bool): Whether to create and save real-time animations
                                  of the sampling evolution. Animations are saved as
                                  MP4 files with filenames including the true parameter value.
        
        Returns:
            np.ndarray: Array of accepted parameter samples after burn-in with shape
                       (n_params, n_samples_post_burnin). Only samples after the burn-in
                       period are returned for posterior analysis.
        
        Algorithm Details:
            - Proposal generation: θ_new ~ N(θ_current, V_current)
            - Acceptance rate tracking for diagnostic purposes
            - Adaptive covariance updates every 'adapt_interval' samples
            - Hierarchical variance updates at each iteration
            - Burn-in removal for final sample return
        
        Animation Features (if MAKE_ANIMATIONS=True):
            - Real-time plot of parameter evolution
            - Automatic axis scaling based on sample range
            - Saved as MP4 with 30 fps using ffmpeg
            - Filename includes true parameter value for identification
        
        Diagnostic Output:
            - Sample index and acceptance status for each iteration
            - Generated sample values for monitoring
            - Final acceptance ratio for algorithm assessment
        
        Note:
            - Burn-in period is automatically set to nsamples/2
            - Failed covariance updates are caught and ignored
            - Animation requires matplotlib and ffmpeg
            - Memory is managed by closing figure after animation
        
        Performance Tips:
            - Set adapt_interval appropriately (typically 10-50)
            - Monitor acceptance ratio (target ~20-50%)
            - Use sufficient burn-in for convergence
            - Consider thinning for large sample sizes
        
        Example:
            >>> # Run MCMC with animations
            >>> samples = mcmc.sample(MAKE_ANIMATIONS=True)
            >>> print(f"Collected {samples.shape[1]} post-burn-in samples")
            >>> print(f"Posterior mean: {np.mean(samples, axis=1):.4f}")
            >>> print(f"Posterior std: {np.std(samples, axis=1):.4f}")
        
        Returns:
            The method returns samples from the posterior distribution that can be used for:
            - Parameter estimation (posterior mean, median)
            - Uncertainty quantification (credible intervals)
            - Model comparison (marginal likelihood estimation)
            - Diagnostic analysis (convergence assessment)
        """
        # Compute initial covariance
        self.compute_initial_covariance()
        
        qparams = np.copy(np.array([[self.qstart]])) # Array of sampled parameters
        Vold = np.copy(self.Vstart) # Covariance matrix of previously sampled parameters
        SSqprev = self.SSqcalc(qparams) # Squared error of previously sampled parameters
        iaccept = 0 # Counter for accepted samples

        if MAKE_ANIMATIONS:
            # Animation setup
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            # Set title and labels
            ax.set_title(f"MCMC Sampling Evolution for dc = {self.dc_true:.2f} as True value")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Sample Value")

            def init():
                ax.set_xlim(0, self.nsamples)  # Set x-axis limits to the number of samples
                # Set y-axis limits to a range that covers the expected values of qparams
                ax.set_ylim(np.min(qparams) - 1, np.max(qparams) + 1) 
                return line,

            def update(frame):
                # Update the line data to reflect the current state of qparams
                # x-axis: indices from 0 to frame, y-axis: qparams values up to current frame
                line.set_data(np.arange(frame), qparams[0, :frame])
                return line,

            anim = FuncAnimation(fig, update, frames=self.nsamples, init_func=init, blit=True)

        for isample in np.arange(self.nsamples):
            # Sample new parameters from a normal distribution 
            # with mean being the last element of qparams
            q_new = np.reshape(np.random.multivariate_normal(qparams[:,-1],Vold),(-1,1)) 

            # Accept or reject the new sample based on the Metropolis-Hastings acceptance rule
            accept,SSqnew = self.acceptreject(q_new,SSqprev,self.std2[-1])

            # Print some diagnostic information
            print(isample,accept)
            print("Generated Sample ---- ", q_new.item())
            # print("Generated Sample ---- ",np.asscalar(q_new))

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

        if MAKE_ANIMATIONS:
            # Save the animation with the extracted dc_value in the filename
            filename = f'mcmc_animation_dc_{self.dc_true:.2f}.mp4'  
            anim.save(filename, fps=30, writer='ffmpeg')

            # Close the figure to free up memory
            plt.close(fig)

        # Return accepted samples
        return qparams[:,self.nburn:]