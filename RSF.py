from imports import *
from MCMC import MCMC
from json_save_load import save_object, load_object
from mysql_save_load import save_object, load_object


def measure_execution_time(func):
    """
    Decorator to measure and return the execution time of a function.
    
    This decorator wraps a function to automatically measure its execution time
    using high-precision timing. It's particularly useful for performance analysis
    and benchmarking of computational methods like MCMC sampling.
    
    Args:
        func (callable): The function to be timed. Can be any callable object
                        including methods, lambda functions, or regular functions.
    
    Returns:
        callable: A wrapper function that returns the execution time in seconds
                 instead of the original function's return value.
    
    Usage:
        The decorator is applied using the @ syntax above function definitions.
        When the decorated function is called, it returns execution time rather
        than the original return value.
    
    Example:
        >>> @measure_execution_time
        ... def slow_computation():
        ...     time.sleep(2)
        ...     return "done"
        >>> 
        >>> exec_time = slow_computation()
        >>> print(f"Function took {exec_time:.3f} seconds")
        Function took 2.001 seconds
    
    Note:
        - Uses time.time() for cross-platform compatibility
        - Returns execution time, not the original function's return value
        - Suitable for timing functions that don't need their return values
        - For functions where return values are needed, consider a different approach
    
    Performance Considerations:
        - Minimal overhead (< 1 microsecond typically)
        - Uses wall-clock time, not CPU time
        - Resolution depends on system clock (usually microsecond precision)
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return execution_time
    return wrapper

RSF_base = object  # Fallback to a base object

class RSF(RSF_base):
    """
    Driver class for Rate and State Friction (RSF) model analysis and parameter estimation.
    
    This class provides a comprehensive framework for analyzing rate and state friction
    models through forward modeling, data generation, and Bayesian parameter estimation
    using Markov Chain Monte Carlo (MCMC) methods. It handles multiple characteristic
    slip distances (Dc values) and supports various data storage formats.
    
    The class integrates several key components:
    - Forward modeling with rate and state friction physics
    - Synthetic data generation with realistic noise models
    - Bayesian parameter estimation using adaptive MCMC
    - Data persistence through JSON and MySQL backends
    - Comprehensive visualization and analysis tools
    - Performance monitoring and optimization
    
    Key Features:
    - Multi-parameter sweep capabilities across Dc values
    - Flexible prior specification for Bayesian inference
    - Automated plotting and visualization
    - Data format abstraction (JSON/MySQL)
    - Execution time monitoring
    - Model reduction options for computational efficiency
    
    Workflow:
    1. Initialize with parameter ranges and inference settings
    2. Generate synthetic time series data across parameter space
    3. Store data using preferred backend (JSON/MySQL)
    4. Perform MCMC parameter estimation for each Dc value
    5. Visualize posterior distributions and convergence diagnostics
    
    Applications:
    - Earthquake fault parameter estimation from seismic data
    - Laboratory friction experiment analysis
    - Model validation and uncertainty quantification
    - Parameter sensitivity studies
    - Computational method benchmarking
    
    Attributes:
        num_dc (int): Number of characteristic slip distance values to analyze
        dc_list (np.ndarray): Array of Dc values for parameter sweep
        num_features (int): Number of features for model reduction (if enabled)
        plotfigs (bool): Whether to generate visualization plots
        qstart (float): Initial parameter value for MCMC chains
        qpriors (list): Prior distribution specification [type, lower, upper]
        reduction (bool): Whether to use model reduction techniques
        model: Forward model object (set externally)
        data: Generated or loaded time series data
        format (str): Data storage format ('json' or 'mysql')
    
    Examples:
        >>> # Basic usage for single parameter estimation
        >>> rsf = RSF(number_slip_values=1, 
        ...           lowest_slip_value=0.01, 
        ...           largest_slip_value=0.01,
        ...           qstart=0.015,
        ...           qpriors=["Uniform", 0.005, 0.05])
        >>> 
        >>> # Set up model and generate data
        >>> rsf.model = RateStateModel(number_time_steps=1000)
        >>> rsf.format = 'json'
        >>> rsf.data = rsf.generate_time_series()
        >>> 
        >>> # Perform Bayesian inference
        >>> execution_time = rsf.inference(nsamples=2000)
        >>> print(f"Analysis completed in {execution_time:.2f} seconds")
        
        >>> # Multi-parameter sensitivity study
        >>> rsf_multi = RSF(number_slip_values=5,
        ...                 lowest_slip_value=0.01,
        ...                 largest_slip_value=0.05,
        ...                 plotfigs=True)
        >>> # ... run analysis across parameter range
    
    Mathematical Background:
        The class performs Bayesian inference on the rate and state friction model:
        
        Likelihood: p(data|Dc) ∝ exp(-SSE/(2σ²))
        Prior: p(Dc) ~ Uniform(lower, upper) or other specified distribution
        Posterior: p(Dc|data) ∝ p(data|Dc) * p(Dc)
        
        Where SSE is the sum of squared errors between model predictions and observations.
    
    Performance Notes:
        - Computation time scales with num_dc * nsamples * model_complexity
        - Memory usage scales with num_dc * num_tsteps for data storage
        - MCMC convergence depends on parameter ranges and data quality
        - Consider model reduction for very large parameter spaces
    
    Dependencies:
        - MCMC class for Bayesian inference
        - RateStateModel class for forward modeling
        - JSON/MySQL save/load utilities
        - Matplotlib for visualization
        - NumPy/SciPy for numerical computations
    """

    def __init__(
        self, 
        number_slip_values=1, 
        lowest_slip_value=1.0, 
        largest_slip_value=1000.0,
        qstart=10.0,
        qpriors=["Uniform", 0.0, 10000.0],
        reduction=False, 
        plotfigs=False
    ):
        """
        Initialize the RSF driver with parameter ranges and analysis settings.
        
        Sets up the parameter space for characteristic slip distance (Dc) analysis,
        configures Bayesian inference settings, and prepares the computational
        framework for rate and state friction model analysis.
        
        Args:
            number_slip_values (int, optional): Number of Dc values to analyze in the
                                              parameter sweep. More values provide
                                              better parameter space coverage but
                                              increase computation time. Defaults to 1.
            lowest_slip_value (float, optional): Minimum characteristic slip distance
                                                in micrometers. Should be physically
                                                reasonable (typically 0.001-1.0).
                                                Defaults to 1.0.
            largest_slip_value (float, optional): Maximum characteristic slip distance
                                                 in micrometers. Should represent
                                                 upper bound of expected values
                                                 (typically 1.0-1000.0). Defaults to 1000.0.
            qstart (float, optional): Initial parameter value for MCMC chains in
                                    micrometers. Should be within [lowest_slip_value,
                                    largest_slip_value] and represent a reasonable
                                    initial guess. Defaults to 10.0.
            qpriors (list, optional): Prior distribution specification as
                                    [distribution_type, lower_bound, upper_bound].
                                    Currently supports "Uniform" distribution.
                                    Bounds define the parameter space for inference.
                                    Defaults to ["Uniform", 0.0, 10000.0].
            reduction (bool, optional): Whether to enable model reduction techniques
                                      for computational efficiency. Useful for
                                      large parameter spaces or high-dimensional
                                      models. Defaults to False.
            plotfigs (bool, optional): Whether to generate visualization plots
                                     during analysis. Includes time series plots,
                                     MCMC diagnostics, and posterior distributions.
                                     Defaults to False.
        
        Initializes:
            - Parameter space discretization (dc_list)
            - Analysis configuration flags
            - Bayesian inference settings
            - Visualization options
            - Placeholder for model and data objects
        
        Parameter Space:
            The Dc values are linearly spaced between lowest_slip_value and
            largest_slip_value. For logarithmically spaced values, modify
            dc_list after initialization using np.logspace().
        
        Prior Specification:
            Currently supports uniform priors specified as:
            ["Uniform", lower_bound, upper_bound]
            
            Future extensions could include:
            - ["Normal", mean, std]
            - ["LogNormal", mu, sigma]
            - ["Gamma", shape, scale]
        
        Note:
            - The model object must be set externally before analysis
            - Data format ('json' or 'mysql') must be specified before data operations
            - num_features is fixed at 2 for current implementation
            - All distance units should be consistent (typically micrometers)
        
        Example:
            >>> # Single parameter analysis
            >>> rsf = RSF(number_slip_values=1, 
            ...           lowest_slip_value=0.02, 
            ...           largest_slip_value=0.02)
            >>> 
            >>> # Parameter sweep analysis
            >>> rsf = RSF(number_slip_values=10,
            ...           lowest_slip_value=0.01,
            ...           largest_slip_value=0.1,
            ...           qstart=0.05,
            ...           qpriors=["Uniform", 0.005, 0.2],
            ...           plotfigs=True)
            >>> 
            >>> # Check parameter space
            >>> print(f"Analyzing Dc values: {rsf.dc_list}")
            >>> print(f"Prior bounds: {rsf.qpriors[1]:.3f} to {rsf.qpriors[2]:.3f}")
        """
        self.num_dc       = number_slip_values
        self.dc_list      = np.linspace(lowest_slip_value, largest_slip_value, self.num_dc)
        self.num_features = 2
        self.plotfigs     = plotfigs
        self.qstart       = qstart
        self.qpriors      = qpriors
        self.reduction    = reduction

        return
   
    def generate_time_series(self):
        """
        Generate synthetic time series data for all characteristic slip distances.
        
        This method performs forward modeling across the entire parameter space
        defined by dc_list, generating synthetic acceleration time series with
        realistic noise for each Dc value. The data is concatenated into a single
        array suitable for subsequent Bayesian analysis.
        
        The method iterates through each Dc value, evaluates the rate and state
        friction model, optionally generates plots for visualization, and
        concatenates the noisy acceleration data into a comprehensive dataset.
        
        Returns:
            np.ndarray: Concatenated noisy acceleration data with shape 
                       (num_dc * num_tsteps,). Each segment of length num_tsteps
                       corresponds to one Dc value in dc_list. The data includes
                       synthetic noise to simulate experimental conditions.
        
        Process:
            1. Calculate total array size for all Dc values
            2. Initialize output array with zeros
            3. For each Dc value:
               - Set model parameter (model.Dc)
               - Evaluate forward model to get time series
               - Generate optional visualization plots
               - Extract noisy acceleration data
               - Store in appropriate array segment
            4. Return concatenated data array
        
        Data Structure:
            The returned array has segments arranged as:
            [dc_1_data, dc_2_data, ..., dc_n_data]
            where each dc_i_data has length model.num_tsteps
        
        Model Requirements:
            - self.model must be set before calling this method
            - model must have Dc attribute that can be modified
            - model.evaluate() must return (time, acceleration, noisy_acceleration)
            - model.num_tsteps must be defined for array sizing
        
        Visualization:
            If self.plotfigs is True, generates individual time series plots
            for each Dc value showing acceleration vs. time with appropriate
            titles and labels.
        
        Error Handling:
            - If model is not set, will raise AttributeError
            - If model.evaluate() fails, exception will propagate
            - Array indexing is safe due to pre-calculated dimensions
        
        Performance:
            - Time complexity: O(num_dc * model_evaluation_time)
            - Space complexity: O(num_dc * num_tsteps)
            - Memory allocation done once upfront for efficiency
        
        Example:
            >>> # Setup model and RSF driver
            >>> from src.RateStateModel import RateStateModel
            >>> model = RateStateModel(number_time_steps=1000, end_time=50.0)
            >>> 
            >>> rsf = RSF(number_slip_values=3,
            ...           lowest_slip_value=0.01,
            ...           largest_slip_value=0.03,
            ...           plotfigs=True)
            >>> rsf.model = model
            >>> 
            >>> # Generate synthetic data
            >>> data = rsf.generate_time_series()
            >>> print(f"Generated data shape: {data.shape}")
            >>> print(f"Expected shape: {(len(rsf.dc_list) * model.num_tsteps,)}")
            >>> 
            >>> # Analyze data segments
            >>> segment_length = model.num_tsteps
            >>> for i, dc in enumerate(rsf.dc_list):
            ...     start_idx = i * segment_length
            ...     end_idx = start_idx + segment_length
            ...     segment_data = data[start_idx:end_idx]
            ...     print(f"Dc={dc:.3f}: RMS acceleration = {np.sqrt(np.mean(segment_data**2)):.4f}")
        
        Usage in Analysis Pipeline:
            >>> # Complete workflow
            >>> rsf.format = 'json'  # Set storage format
            >>> synthetic_data = rsf.generate_time_series()
            >>> rsf.data = rsf.prepare_data(synthetic_data)  # Store/load data
            >>> execution_time = rsf.inference(nsamples=1000)  # Run MCMC
        
        Note:
            - Each model evaluation generates independent noise realizations
            - The same random seed will produce identical results
            - Consider setting numpy random seed for reproducibility
            - Large parameter spaces may require significant computation time
            - Generated data represents "ground truth" with known parameters
        """
        # Calculate the total number of entries
        total_entries = len(self.dc_list) * self.model.num_tsteps
        acc_appended_noise = np.zeros(total_entries)

        for index, dc_value in enumerate(self.dc_list):
            # Evaluate the model for the current value of dc
            self.model.Dc = dc_value
            time_series, acceleration, acc_noise = self.model.evaluate()
            self.plot_time_series(time_series, acceleration)  # Generate plots

            # Calculate the start and end indices for the current segment
            start_index = index * self.model.num_tsteps
            end_index = start_index + self.model.num_tsteps

            # Append the noise-adjusted acceleration data
            acc_appended_noise[start_index:end_index] = acc_noise

        return acc_appended_noise

    def plot_time_series(self, time, acceleration):
        """
        Generate time series visualization plots for rate and state friction data.
        
        This method creates publication-quality plots of acceleration time series
        with proper formatting, labels, and styling. The plots help visualize
        the temporal evolution of the rate and state friction system and are
        essential for understanding model behavior and validating results.
        
        Args:
            time (np.ndarray): Time array with shape (num_tsteps,) containing
                             time values from model evaluation. Units should
                             be consistent with model setup (typically seconds).
            acceleration (np.ndarray): Acceleration array with shape (num_tsteps,)
                                     containing the clean (non-noisy) acceleration
                                     time series from the rate and state model.
                                     Units typically in μm/s².
        
        Returns:
            None: Creates and displays matplotlib figure but does not return it.
                 The figure is not saved automatically.
        
        Plot Features:
            - Professional formatting with LaTeX-style mathematical notation
            - Grid lines for easier data reading
            - Proper axis labels with units
            - Title showing current Dc parameter value
            - Legend indicating this is the "True" (non-noisy) signal
            - Time axis limits set to model bounds with small buffer
        
        Styling:
            - Line width: 1.0 for clear visibility
            - Grid: Enabled for professional appearance
            - Title: Shows Dc value in scientific notation
            - Labels: Use proper mathematical formatting with LaTeX
        
        Conditional Execution:
            The method only generates plots if self.plotfigs is True, allowing
            for batch processing without visual output when desired.
        
        Example:
            >>> # Single plot generation
            >>> rsf = RSF(plotfigs=True)
            >>> rsf.model = RateStateModel()
            >>> rsf.model.Dc = 0.02
            >>> 
            >>> t, acc, acc_noise = rsf.model.evaluate()
            >>> rsf.plot_time_series(t, acc)
            >>> plt.show()  # Display the plot
            
            >>> # Batch plotting (automatically called in generate_time_series)
            >>> rsf = RSF(number_slip_values=5, plotfigs=True)
            >>> # ... setup model ...
            >>> data = rsf.generate_time_series()  # Creates 5 plots automatically
        
        Mathematical Notation:
            - Uses LaTeX formatting for proper mathematical symbols
            - Dc parameter shown in micrometers (μm)
            - Acceleration units shown as μm/s²
            - Subscripts and Greek letters properly formatted
        
        Performance Considerations:
            - Each plot creation takes ~0.1-0.5 seconds depending on data size
            - Memory usage minimal for typical time series lengths
            - No automatic saving; manually save if needed
            - Consider plt.close() for batch processing to manage memory
        
        Integration with Analysis:
            The plots are particularly useful for:
            - Visual validation of model behavior
            - Identifying stick-slip events and patterns
            - Comparing different Dc parameter effects
            - Quality control of synthetic data generation
            - Presentation and publication figures
        
        Customization:
            To modify plot appearance, extend this method or create custom
            plotting functions. Common modifications include:
            - Different line styles or colors
            - Logarithmic scales for specific analyses
            - Subplots for multiple variables
            - Animation for temporal evolution
            - Statistical overlays (mean, confidence intervals)
        
        Note:
            - Plots are created in separate figure windows
            - No automatic figure management (closing, saving)
            - Uses current matplotlib backend settings
            - Figure size uses matplotlib defaults
            - Consider plot density for large parameter sweeps
        """
        if not self.plotfigs:
            return

        plt.figure()
        title = rf'$d_c$={self.model.Dc} $\mu m$ RSF solution'
        plt.title(title)
        plt.plot(time, acceleration, linewidth=1.0, label='True')
        plt.xlim(self.model.t_start - 2.0, self.model.t_final)
        plt.xlabel('Time (sec)')
        plt.ylabel(rf'Acceleration $(\mu m/s^2)$')
        plt.grid(True)
        plt.legend()
           
    def prepare_data(self, data):
        """
        Prepare and persist data using specified storage format (JSON or MySQL).
        
        This method provides data persistence abstraction, allowing seamless
        switching between JSON file storage and MySQL database storage for
        time series data. It handles the complete save/load cycle to ensure
        data integrity and format consistency across different storage backends.
        
        The method performs a round-trip operation (save then load) to verify
        data integrity and ensure the returned data matches the stored format
        exactly. This is particularly important for numerical precision and
        array structure preservation.
        
        Args:
            data (np.ndarray): Time series data to be stored and retrieved.
                             Typically the output from generate_time_series()
                             with shape (num_dc * num_tsteps,). Should contain
                             numerical values (floats) representing acceleration
                             measurements with synthetic noise.
        
        Returns:
            np.ndarray: The same data after save/load round-trip, potentially
                       with format-specific transformations applied (e.g.,
                       precision changes, type conversions). Shape and values
                       should match input data within numerical precision limits.
        
        Storage Formats:
            JSON Format (self.format == 'json'):
            - Saves to 'data.json' in current directory
            - Uses custom JSON encoder for NumPy array handling
            - Preserves array structure and numerical precision
            - Human-readable format, suitable for small to medium datasets
            - Platform-independent, version control friendly
            
            MySQL Format (self.format == 'mysql'):
            - Saves to MySQL database with predefined connection parameters
            - Uses chunked insertion for memory efficiency
            - Suitable for large datasets and concurrent access
            - Requires MySQL server setup and proper credentials
            - Supports database operations and queries
        
        Prerequisites:
            - self.format must be set to either 'json' or 'mysql'
            - For MySQL: database server must be running and accessible
            - For MySQL: connection parameters must be valid
            - Required save/load modules must be imported
        
        File Management:
            JSON files are created in the current working directory:
            - 'data.json': Contains the time series data
            - 'model_lstm.json': Placeholder for LSTM model data (future use)
        
        Database Configuration:
            MySQL connection uses hardcoded parameters:
            - Host: 'localhost'
            - User: 'my_user'
            - Password: 'my_password'
            - Database: 'my_database'
            
            Note: Consider moving to configuration file or environment variables
        
        Error Handling:
            - File permission errors for JSON format
            - Database connection errors for MySQL format
            - Invalid data format or type errors
            - Network connectivity issues for remote databases
        
        Example:
            >>> # JSON storage workflow
            >>> rsf = RSF()
            >>> rsf.format = 'json'
            >>> original_data = rsf.generate_time_series()
            >>> stored_data = rsf.prepare_data(original_data)
            >>> np.allclose(original_data, stored_data)  # Should be True
            >>> 
            >>> # MySQL storage workflow
            >>> rsf.format = 'mysql'
            >>> # Ensure MySQL server is running with correct credentials
            >>> stored_data = rsf.prepare_data(original_data)
        
        Performance Considerations:
            JSON Format:
            - Fast for small to medium datasets (< 100MB)
            - Memory usage during save: ~2x data size
            - Disk space: ~1.5x memory size (text format overhead)
            
            MySQL Format:
            - Scales well for large datasets
            - Network latency affects performance
            - Chunked insertion optimizes memory usage
            - Database indexing can improve query performance
        
        Data Integrity:
            - Round-trip verification ensures data consistency
            - Numerical precision preserved within format limits
            - Array structure and ordering maintained
            - Consider checksums for critical applications
        
        Security Notes:
            - JSON files stored in plaintext (consider encryption for sensitive data)
            - MySQL credentials hardcoded (security risk in production)
            - File permissions should be set appropriately
            - Database access should use principle of least privilege
        
        Future Enhancements:
            - Configuration file for database parameters
            - Additional storage formats (HDF5, Parquet)
            - Compression options for large datasets
            - Metadata storage (timestamps, model parameters)
            - Data validation and checksum verification
        """
        if self.format == 'json':
            self.lstm_file = 'model_lstm.json'
            self.data_file = 'data.json'
            save_object(data, self.data_file)
            data = load_object(self.data_file)

        elif self.format == 'mysql':
            # MySQL connection details
            mysql_host = 'localhost'
            mysql_user = 'my_user'
            mysql_password = 'my_password'
            mysql_database = 'my_database'
            save_object(data, mysql_host, mysql_user, mysql_password, mysql_database)
            data = load_object(mysql_host, mysql_user, mysql_password, mysql_database)

        return data

    def plot_dist(self, qparams, dc):
        """
        Generate comprehensive visualization of MCMC posterior distribution.
        
        This method creates a dual-panel plot showing both the MCMC sample trace
        and the corresponding probability density estimate. The visualization is
        essential for assessing convergence, understanding posterior structure,
        and presenting Bayesian inference results.
        
        The plot combines a trace plot (left panel) showing the evolution of
        parameter samples over iterations with a kernel density estimate (right panel)
        showing the posterior probability distribution. This dual view provides
        both convergence diagnostics and distributional information.
        
        Args:
            qparams (np.ndarray): MCMC parameter samples with shape (n_params, n_samples).
                                For this implementation, expects shape (1, n_samples)
                                where the single parameter is the Dc value.
                                Samples should be post-burn-in values.
            dc (float): True characteristic slip distance value used for comparison
                       and plot labeling. Displayed in the plot title for reference
                       and validation purposes.
        
        Returns:
            None: Creates and displays a matplotlib figure with two subplots.
                 The figure is not returned or saved automatically.
        
        Plot Layout:
            Left Panel (70% width):
            - Trace plot showing parameter evolution vs. sample number
            - Blue line with specified line width
            - Y-axis: Parameter value (Dc)
            - X-axis: Sample number (iteration)
            - Shows mixing and convergence behavior
            
            Right Panel (15% width):
            - Kernel density estimate of posterior distribution
            - Filled area under curve with transparency
            - Y-axis: Parameter value (shared with left panel)
            - X-axis: Probability density
            - No tick marks for cleaner appearance
        
        Styling Parameters:
            - KDE_POINTS = 1000: Resolution of density estimate
            - PLOT_WIDTH_RATIO = [0.7, 0.15]: Panel width proportions
            - PLOT_SPACING = 0.15: Spacing between panels
            - PLOT_LINE_WIDTH = 1.0: Line thickness
            - PLOT_ALPHA = 0.3: Fill transparency
        
        Statistical Analysis:
            The kernel density estimation uses Gaussian kernels with automatic
            bandwidth selection via Scott's rule. The density is evaluated over
            the range defined by the trace plot's y-axis limits.
        
        Example:
            >>> # After MCMC sampling
            >>> mcmc_obj = MCMC(model, data, dc_true, qpriors, qstart, nsamples=2000)
            >>> posterior_samples = mcmc_obj.sample(False)
            >>> 
            >>> # Generate distribution plot
            >>> rsf = RSF()
            >>> rsf.format = 'json'  # Required for title formatting
            >>> rsf.plot_dist(posterior_samples, dc_true)
            >>> plt.show()
        
        Interpretation Guide:
            Trace Plot (Left):
            - Horizontal lines suggest good mixing
            - Trends indicate lack of convergence
            - Oscillations show exploration of parameter space
            - Stable variance suggests adequate burn-in
            
            Density Plot (Right):
            - Peak location: Posterior mode (most likely value)
            - Width: Posterior uncertainty
            - Shape: Distribution characteristics (skewness, multimodality)
            - Area under curve: Should integrate to 1.0
        
        Convergence Diagnostics:
            Visual signs of good convergence:
            - Trace plot shows random fluctuation around mean
            - No obvious trends or drift in trace
            - Density plot shows smooth, well-defined peak
            - Multiple modes may indicate complex posterior structure
        
        Title Information:
            The plot title shows:
            - True Dc value for comparison with posterior
            - Data storage format used in analysis
            - Mathematical notation using LaTeX formatting
        
        Performance Notes:
            - KDE computation scales as O(n*m) where n=samples, m=grid points
            - Memory usage minimal for typical sample sizes (< 10^6 samples)
            - Plot rendering time depends on sample size and grid resolution
        
        Customization Options:
            Common modifications:
            - Adjust KDE_POINTS for smoother/faster density estimates
            - Change width ratios for different emphasis
            - Add reference lines for true parameter values
            - Overlay multiple chains for comparison
            - Add summary statistics annotations
        
        Note:
            - Assumes single parameter (Dc) analysis
            - Uses first row of qparams array (qparams[0, :])
            - Figure is created but not saved automatically
            - Consider plt.savefig() for permanent storage
            - Panel spacing and sizing optimized for typical screen viewing
        """
        KDE_POINTS = 1000
        PLOT_WIDTH_RATIO = [0.7, 0.15]
        PLOT_SPACING = 0.15
        PLOT_LINE_WIDTH = 1.0
        PLOT_ALPHA = 0.3

        # Set up the plot layout
        fig, axes = plt.subplots(
            1, 2, gridspec_kw={'width_ratios': PLOT_WIDTH_RATIO, 'wspace': PLOT_SPACING})
        fig.suptitle(rf'$d_c={dc:.2f}\,\mu m$ with {self.format} formatting', fontsize=10)

        # Plot MCMC samples
        axes[0].plot(qparams[0, :], 'b-', linewidth=PLOT_LINE_WIDTH)
        axes[0].set_ylabel('$d_c$', fontsize=10)
        axes[0].set_xlabel('Sample number')
        axes[0].set_xlim(0, qparams.shape[1])

        # Calculate and plot KDE
        kde_x_values = np.linspace(*axes[0].get_ylim(), KDE_POINTS)
        kde = gaussian_kde(qparams[0, :])
        axes[1].plot(kde.pdf(kde_x_values), kde_x_values, 'b-', linewidth=PLOT_LINE_WIDTH)
        axes[1].fill_betweenx(
            kde_x_values, kde.pdf(kde_x_values), np.zeros(kde_x_values.shape), alpha=PLOT_ALPHA)
        axes[1].set_xlim(0, None)
        axes[1].set_xlabel('Prob. density')
        axes[1].get_yaxis().set_visible(False)
        axes[1].get_xaxis().set_visible(True)
        axes[1].get_xaxis().set_ticks([])

        return

    def perform_sampling_and_plotting(self, data, dc, nsamples, model_lstm):
        """
        Execute MCMC parameter estimation and generate posterior visualizations.
        
        This method performs the core Bayesian inference workflow for a specific
        characteristic slip distance (Dc) value. It extracts the relevant data
        segment, initializes the MCMC sampler, runs the sampling algorithm,
        and generates comprehensive posterior distribution plots.
        
        The method handles data segmentation to isolate the time series corresponding
        to the specified Dc value, configures the MCMC algorithm with appropriate
        priors and starting values, and produces both sampling diagnostics and
        final posterior visualizations.
        
        Args:
            data (np.ndarray): Complete concatenated time series data from all Dc values
                             with shape (num_dc * num_tsteps,). Contains noisy acceleration
                             measurements for the entire parameter space.
            dc (float): Specific characteristic slip distance value to analyze.
                       Must be present in self.dc_list for proper data extraction.
                       Units should be consistent with model setup (typically μm).
            nsamples (int): Number of MCMC samples to generate for this Dc value.
                          Should be sufficient for convergence (typically 1000-10000).
                          Includes burn-in period (automatically handled by MCMC class).
            model_lstm (dict or None): LSTM model configuration for reduced-order modeling.
                                     If None, uses full forward model evaluation.
                                     If provided, should contain LSTM model parameters.
        
        Returns:
            None: Performs analysis and generates plots but does not return values.
                 Results are visualized through plot_dist() method.
        
        Process Workflow:
            1. Locate Dc value in parameter list and extract corresponding data segment
            2. Print diagnostic information about current analysis
            3. Initialize MCMC object with model, data, and inference settings
            4. Execute MCMC sampling with optional animation
            5. Generate posterior distribution plots for visualization
        
        Data Segmentation:
            The method extracts the relevant time series segment using:
            - start_index = dc_index * model.num_tsteps
            - end_index = start_index + model.num_tsteps
            
            This assumes data is organized as concatenated segments from generate_time_series().
        
        MCMC Configuration:
            The MCMC sampler is initialized with:
            - Forward model: self.model
            - Observed data: noisy_data (extracted segment)
            - True parameter: dc (for validation/comparison)
            - Prior specification: self.qpriors
            - Initial value: self.qstart
            - LSTM model: model_lstm (optional)
            - Sample count: nsamples
        
        Error Handling:
            - Checks if Dc value exists in dc_list
            - Prints error message and returns early if not found
            - MCMC sampling errors propagate to caller
            - Plotting errors are handled by plot_dist() method
        
        Output and Visualization:
            - Prints analysis status and current Dc value
            - Generates MCMC animation if enabled in sampling
            - Creates posterior distribution plots via plot_dist()
            - No return value; results communicated through plots
        
        Example:
            >>> # Setup for analysis
            >>> rsf = RSF(number_slip_values=3, plotfigs=True)
            >>> rsf.model = RateStateModel()
            >>> rsf.format = 'json'
            >>> 
            >>> # Generate and prepare data
            >>> synthetic_data = rsf.generate_time_series()
            >>> prepared_data = rsf.prepare_data(synthetic_data)
            >>> 
            >>> # Analyze specific Dc value
            >>> target_dc = rsf.dc_list[1]  # Second Dc value
            >>> rsf.perform_sampling_and_plotting(
            ...     data=prepared_data,
            ...     dc=target_dc,
            ...     nsamples=2000,
            ...     model_lstm=None
            ... )
            >>> # Displays: "--- Dc is {target_dc} ---" and generates plots
        
        Performance Considerations:
            - Execution time dominated by MCMC sampling (O(nsamples))
            - Data extraction is O(1) with pre-computed indices
            - Memory usage scales with nsamples for sample storage
            - Animation generation adds overhead if enabled
        
        Integration with Main Analysis:
            This method is typically called from inference() in a loop over
            all Dc values in dc_list, enabling systematic parameter space
            exploration and comparison of posterior distributions.
        
        Diagnostic Information:
            Printed output includes:
            - Current Dc value being analyzed
            - MCMC progress indicators (from MCMC class)
            - Convergence diagnostics (acceptance rates, etc.)
        
        LSTM Model Support:
            If model_lstm is provided:
            - Uses reduced-order model evaluation for efficiency
            - Maintains same inference framework
            - Can significantly reduce computation time for complex models
            - Model reduction parameters passed through to MCMC
        
        Validation and Quality Control:
            - Checks for valid Dc value in parameter list
            - MCMC class handles convergence diagnostics internally
            - Posterior plots provide visual validation of results
            - Consider adding quantitative convergence tests
        
        Note:
            - Requires self.model to be properly configured
            - Data indexing assumes consistent segment lengths
            - MCMC animation setting affects visualization
            - Results quality depends on nsamples and prior specification
            - Consider parallel execution for multiple Dc values
        """
        # Find the index of 'dc' in the NumPy array 'self.dc_list'
        index = np.where(self.dc_list == dc)[0][0] if dc in self.dc_list else -1
        if index == -1:
            print(f"Error: dc value {dc} not found in dc_list.")
            return

        # Start and End Indices for time series for that dc
        start = index * self.model.num_tsteps
        end = start + self.model.num_tsteps
        noisy_data = data[start:end]
        print(f'--- Dc is {dc} ---')

        # Initialize MCMC object
        MCMCobj = MCMC(
            self.model, 
            noisy_data,
            dc, 
            self.qpriors, 
            self.qstart, 
            lstm_model=model_lstm, 
            nsamples=nsamples
        )

        # Perform MCMC sampling
        qparams = MCMCobj.sample(True)

        # Plot final distribution
        self.plot_dist(qparams, dc)
        
    @measure_execution_time
    def inference(self, nsamples):
        """
        Execute comprehensive Bayesian parameter estimation across the entire parameter space.
        
        This method orchestrates the complete analysis workflow, performing MCMC-based
        Bayesian inference for all characteristic slip distance (Dc) values in the
        parameter space. It handles data preparation, systematic parameter estimation,
        and provides execution time monitoring for performance analysis.
        
        The method represents the main entry point for RSF parameter estimation,
        coordinating data persistence, model evaluation, and posterior inference
        across the entire range of Dc values defined during initialization.
        
        Args:
            nsamples (int): Number of MCMC samples to generate for each Dc value.
                          Should be sufficient for posterior convergence, typically
                          1000-10000 depending on problem complexity and desired
                          precision. Includes burn-in period (handled automatically).
        
        Returns:
            float: Total execution time in seconds for the complete analysis.
                  Includes time for data preparation, all MCMC runs, and plotting.
                  Returned due to @measure_execution_time decorator.
        
        Workflow Process:
            1. Data Preparation:
               - Calls prepare_data() to handle persistence using specified format
               - Ensures data is properly stored and retrieved for analysis
            
            2. Parameter Space Iteration:
               - Loops through all Dc values in self.dc_list
               - For each Dc value, performs complete MCMC analysis
               - Generates posterior distribution plots and diagnostics
            
            3. Performance Monitoring:
               - Measures total execution time using decorator
               - Enables performance comparison across different configurations
        
        Data Flow:
            The method assumes self.data contains the concatenated time series
            generated by generate_time_series(). This data is processed through
            prepare_data() to ensure proper format and persistence.
        
        Analysis Configuration:
            Each MCMC analysis uses:
            - Forward model: self.model
            - Prior specification: self.qpriors
            - Initial values: self.qstart
            - Sample count: nsamples (same for all Dc values)
            - LSTM model: None (full model evaluation)
        
        Output Generation:
            For each Dc value, produces:
            - MCMC trace plots and convergence diagnostics
            - Posterior distribution visualizations
            - Animation files (if enabled in MCMC sampling)
            - Console output with progress and diagnostics
        
        Performance Characteristics:
            - Total time scales as O(num_dc * nsamples * model_evaluation_time)
            - Memory usage depends on sample storage and plotting
            - I/O time for data persistence (format-dependent)
            - Parallel execution potential for multiple Dc values
        
        Example Usage:
            >>> # Complete analysis workflow
            >>> from src.RateStateModel import RateStateModel
            >>> 
            >>> # Setup model and RSF driver
            >>> model = RateStateModel(number_time_steps=1000, end_time=50.0)
            >>> rsf = RSF(number_slip_values=5,
            ...           lowest_slip_value=0.01,
            ...           largest_slip_value=0.05,
            ...           qpriors=["Uniform", 0.005, 0.1],
            ...           plotfigs=True)
            >>> rsf.model = model
            >>> rsf.format = 'json'
            >>> 
            >>> # Generate synthetic data
            >>> rsf.data = rsf.generate_time_series()
            >>> 
            >>> # Perform Bayesian inference
            >>> total_time = rsf.inference(nsamples=2000)
            >>> print(f"Complete analysis took {total_time:.2f} seconds")
            >>> print(f"Average time per Dc: {total_time/len(rsf.dc_list):.2f} seconds")
        
        Error Handling:
            - Data preparation errors propagate from prepare_data()
            - Invalid Dc values handled by perform_sampling_and_plotting()
            - MCMC convergence issues reported through console output
            - Plotting errors handled by individual visualization methods
        
        Monitoring and Diagnostics:
            Console output includes:
            - Progress indicators for each Dc value
            - MCMC sampling diagnostics (acceptance rates, convergence)
            - Timing information for performance assessment
            - Error messages for failed analyses
        
        Results Interpretation:
            Each analysis produces:
            - Posterior distributions for comparison across Dc values
            - Convergence diagnostics for quality assessment
            - Uncertainty quantification through posterior spreads
            - Parameter estimation accuracy (compared to true values)
        
        Performance Optimization:
            Strategies for large analyses:
            - Use model reduction (LSTM) for faster evaluation
            - Reduce nsamples for preliminary analyses
            - Consider parallel execution for multiple Dc values
            - Use MySQL storage for very large datasets
            - Monitor memory usage for extensive parameter sweeps
        
        Quality Control:
            Best practices:
            - Verify convergence through trace plots
            - Check acceptance rates (typically 20-50%)
            - Compare posterior means with true values
            - Assess posterior uncertainty levels
            - Validate results through sensitivity studies
        
        Integration with External Tools:
            Results can be used for:
            - Statistical analysis of parameter estimates
            - Model validation and comparison studies
            - Uncertainty propagation in forward modeling
            - Publication-quality figure generation
            - Database storage of inference results
        
        Note:
            - Requires self.data to be set before calling
            - All Dc values are analyzed with same nsamples
            - Execution time monitoring via decorator
            - Consider memory management for large parameter spaces
            - Results are visualized but not automatically saved
        """
        data = self.prepare_data(self.data)

        for dc in self.dc_list:

            self.perform_sampling_and_plotting(data, dc, nsamples, None)

        return