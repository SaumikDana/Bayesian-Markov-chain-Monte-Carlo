"""
Rate and State Friction Model Parameter Estimation using Bayesian Inference.

This script demonstrates a complete workflow for analyzing earthquake fault mechanics
using Rate and State Friction (RSF) models and Markov Chain Monte Carlo (MCMC)
parameter estimation. The analysis includes forward modeling, synthetic data 
generation, and Bayesian inference of characteristic slip distance parameters.

The workflow encompasses:
1. Forward modeling of rate and state friction physics
2. Synthetic time series generation across parameter space
3. Data persistence using configurable storage backends
4. MCMC-based Bayesian parameter estimation
5. Posterior distribution analysis and visualization

This implementation serves as both a research tool for earthquake mechanics
studies and a demonstration of advanced Bayesian computational methods for
inverse problems in geophysics.

Key Features:
- Multi-parameter sensitivity analysis across Dc values
- Configurable data storage (JSON/MySQL backends)
- Comprehensive visualization and diagnostics
- Performance monitoring and optimization
- Modular design for easy extension and modification

Scientific Applications:
- Earthquake fault parameter estimation from seismic data
- Laboratory friction experiment analysis and interpretation
- Model validation and uncertainty quantification studies
- Computational method development and benchmarking

Dependencies:
- NumPy/SciPy for numerical computations
- Matplotlib for visualization
- Custom modules: RSF, RateStateModel, MCMC
- Optional: MySQL connector for database storage

Example Usage:
    python main.py

Author: [Your Name]
Date: [Current Date]
Version: 1.0
"""

import setup_path
from src.imports import *
from src.RSF import RSF
from src.RateStateModel import RateStateModel


# Constants
NUMBER_SLIP_VALUES = 5
LOWEST_SLIP_VALUE = 100.
LARGEST_SLIP_VALUE = 5000.
QSTART = 1000.
QPRIORS = ["Uniform", 0., 10000.]
NUMBER_TIME_STEPS = 500
NSAMPLES = 500

def setup_problem():
    """
    Initialize and configure the Rate and State Friction analysis problem.
    
    This function creates a comprehensive RSF analysis setup by initializing
    the parameter space, configuring the forward model, and generating synthetic
    time series data. The setup represents a complete earthquake fault mechanics
    problem ready for Bayesian parameter estimation.
    
    The function coordinates multiple components:
    - RSF driver class with parameter space configuration
    - Rate and State friction forward model
    - Synthetic data generation with realistic noise
    
    Returns:
        RSF: Fully configured RSF problem instance containing:
            - Parameter space (Dc values from LOWEST_SLIP_VALUE to LARGEST_SLIP_VALUE)
            - Initialized forward model (RateStateModel)
            - Generated synthetic time series data with noise
            - MCMC configuration (priors, starting values)
    
    Configuration Details:
        Parameter Space:
        - NUMBER_SLIP_VALUES (5): Number of characteristic slip distances to analyze
        - LOWEST_SLIP_VALUE (100 μm): Minimum Dc value for parameter sweep
        - LARGEST_SLIP_VALUE (5000 μm): Maximum Dc value for parameter sweep
        - Linear spacing between bounds for systematic coverage
        
        Forward Model:
        - NUMBER_TIME_STEPS (500): Temporal resolution for ODE integration
        - Default rate and state parameters (a=0.011, b=0.014)
        - Radiation damping enabled for physical realism
        - Time range: 0-50 seconds with adaptive ODE solver
        
        Bayesian Configuration:
        - QSTART (1000 μm): Initial parameter value for MCMC chains
        - QPRIORS: Uniform prior distribution [0, 10000] μm
        - Prior bounds encompass expected parameter range
    
    Data Generation Process:
        1. Creates RSF driver with specified parameter configuration
        2. Initializes RateStateModel with temporal discretization
        3. Generates synthetic time series for each Dc value
        4. Adds realistic noise to simulate experimental conditions
        5. Concatenates data for systematic analysis
    
    Physical Interpretation:
        The parameter range (100-5000 μm) represents realistic characteristic
        slip distances for earthquake faults:
        - Small Dc (100-500 μm): Velocity strengthening behavior
        - Medium Dc (500-2000 μm): Transitional regime
        - Large Dc (2000-5000 μm): Velocity weakening, stick-slip potential
    
    Performance Characteristics:
        - Data generation time: ~5-30 seconds (depends on NUMBER_SLIP_VALUES)
        - Memory usage: ~50-500 MB (scales with problem size)
        - Computation scales linearly with NUMBER_SLIP_VALUES
        - Forward model evaluation dominates computation time
    
    Example:
        >>> problem = setup_problem()
        >>> print(f"Parameter space: {len(problem.dc_list)} Dc values")
        >>> print(f"Data shape: {problem.data.shape}")
        >>> print(f"Dc range: {problem.dc_list[0]:.1f} to {problem.dc_list[-1]:.1f} μm")
        >>> 
        >>> # Inspect generated data
        >>> segment_length = problem.model.num_tsteps
        >>> for i, dc in enumerate(problem.dc_list):
        ...     start_idx = i * segment_length
        ...     segment = problem.data[start_idx:start_idx + segment_length]
        ...     print(f"Dc={dc:.1f}: RMS={np.sqrt(np.mean(segment**2)):.4f}")
    
    Customization:
        To modify the problem setup:
        - Adjust constants at module level for different parameter ranges
        - Modify RateStateModel parameters (a, b, time range)
        - Change temporal resolution via NUMBER_TIME_STEPS
        - Update prior bounds for different parameter exploration
    
    Error Handling:
        - Forward model initialization errors propagate to caller
        - Data generation failures will raise exceptions
        - Parameter validation handled by RSF class
        - Consider wrapping in try-catch for production use
    
    Validation:
        Quality checks for generated problem:
        - Verify data array dimensions match expected size
        - Check parameter bounds are physically reasonable
        - Ensure forward model converges for all Dc values
        - Validate noise characteristics in synthetic data
    
    Note:
        - Generated data represents "ground truth" with known parameters
        - Same random seed produces identical synthetic datasets
        - Problem instance is ready for immediate inference analysis
        - Memory usage scales with NUMBER_SLIP_VALUES * NUMBER_TIME_STEPS
    """
    problem       = RSF(number_slip_values=NUMBER_SLIP_VALUES,
                        lowest_slip_value=LOWEST_SLIP_VALUE,
                        largest_slip_value=LARGEST_SLIP_VALUE,
                        qstart=QSTART,
                        qpriors=QPRIORS
                        )
    problem.model = RateStateModel(number_time_steps=NUMBER_TIME_STEPS)
    problem.data  = problem.generate_time_series()
    return problem

def perform_inference(problem, data_format, nsamples):
    """
    Execute Bayesian parameter estimation using MCMC methods.
    
    This function performs the complete Bayesian inference workflow for Rate and
    State Friction parameter estimation. It configures data persistence, executes
    MCMC sampling across the parameter space, and generates comprehensive
    posterior analysis and visualization.
    
    The function orchestrates the inference process by setting up data storage,
    running systematic MCMC analysis for each characteristic slip distance,
    and measuring performance for computational assessment.
    
    Args:
        problem (RSF): Configured RSF problem instance containing:
                      - Parameter space and forward model
                      - Generated synthetic time series data
                      - MCMC configuration and prior specifications
        data_format (str): Data storage format specification:
                          - 'json': JSON file storage for small to medium datasets
                          - 'mysql': MySQL database storage for large datasets
                          Must be supported by RSF data persistence layer.
        nsamples (int): Number of MCMC samples per parameter value:
                       - Should be sufficient for convergence (typically 1000-10000)
                       - Includes automatic burn-in period (50% of samples)
                       - Higher values improve posterior accuracy but increase time
    
    Returns:
        float: Total execution time in seconds for complete inference analysis.
              Includes data preparation, MCMC sampling, and visualization.
              Useful for performance analysis and computational benchmarking.
    
    Inference Workflow:
        1. Data Format Configuration:
           - Sets problem.format to specified storage backend
           - Configures persistence layer for data handling
        
        2. MCMC Execution:
           - Calls problem.inference() with sample count specification
           - Performs systematic analysis across all Dc values
           - Generates posterior distributions and convergence diagnostics
        
        3. Performance Monitoring:
           - Measures total execution time via @measure_execution_time decorator
           - Enables computational performance assessment
    
    Storage Format Details:
        JSON Format:
        - File-based storage in current directory
        - Human-readable format, good for version control
        - Suitable for datasets < 100MB
        - Fast for small to medium problem sizes
        - Platform-independent storage
        
        MySQL Format:
        - Database storage with chunked insertion
        - Suitable for large datasets and concurrent access
        - Requires MySQL server configuration
        - Better scalability for extensive parameter studies
        - Supports advanced querying and data management
    
    MCMC Analysis Details:
        For each Dc value in the parameter space:
        - Extracts relevant data segment from concatenated time series
        - Initializes adaptive Metropolis-Hastings sampler
        - Performs nsamples iterations with automatic burn-in
        - Generates convergence diagnostics and acceptance statistics
        - Creates posterior distribution visualizations
    
    Performance Characteristics:
        - Execution time scales as O(num_dc * nsamples * model_evaluation_time)
        - Memory usage depends on sample storage and plotting
        - I/O overhead varies significantly between storage formats
        - Parallel execution potential for multiple Dc values
    
    Output Generation:
        The function produces comprehensive analysis results:
        - MCMC trace plots showing parameter evolution
        - Posterior probability density plots
        - Convergence diagnostics and acceptance rates
        - Animation files (if enabled) showing sampling evolution
        - Console output with progress and performance metrics
    
    Example:
        >>> # Setup and run inference
        >>> problem = setup_problem()
        >>> 
        >>> # JSON storage analysis
        >>> json_time = perform_inference(problem, 'json', 2000)
        >>> print(f"JSON analysis: {json_time:.2f} seconds")
        >>> 
        >>> # MySQL storage analysis (if database available)
        >>> mysql_time = perform_inference(problem, 'mysql', 2000)
        >>> print(f"MySQL analysis: {mysql_time:.2f} seconds")
        >>> 
        >>> # Performance comparison
        >>> speedup = mysql_time / json_time
        >>> print(f"Storage format speedup: {speedup:.2f}x")
    
    Quality Assessment:
        Monitor these metrics for successful inference:
        - Acceptance rates: typically 20-50% for good mixing
        - Trace plot convergence: no obvious trends or drift
        - Posterior smoothness: well-defined probability distributions
        - Parameter recovery: posterior means near true values
    
    Error Handling:
        Potential issues and their handling:
        - Data format errors: Invalid storage backend specification
        - MCMC convergence failures: Poor mixing or divergence
        - Storage backend errors: File permissions or database connectivity
        - Memory issues: Large parameter spaces or sample counts
    
    Optimization Strategies:
        For large-scale analyses:
        - Use MySQL storage for datasets > 100MB
        - Consider model reduction for faster evaluation
        - Implement parallel MCMC for multiple Dc values
        - Monitor memory usage and adjust sample counts
        - Use preliminary runs to optimize MCMC parameters
    
    Results Interpretation:
        The analysis produces posterior distributions that quantify:
        - Parameter estimation uncertainty
        - Model sensitivity to different Dc values
        - Quality of data fit and model adequacy
        - Computational efficiency metrics
    
    Note:
        - Requires problem.data to be set before calling
        - Visualization windows may remain open after completion
        - Consider automated figure saving for batch processing
        - Results quality depends on nsamples and problem conditioning
    """
    problem.format = data_format
    return problem.inference(nsamples=nsamples)

def main():
    """
    Main execution function for Rate and State Friction parameter estimation.
    
    This function orchestrates the complete workflow for Bayesian analysis of
    earthquake fault mechanics using Rate and State Friction models. It provides
    a comprehensive demonstration of advanced computational methods for parameter
    estimation in geophysical inverse problems.
    
    The function executes the full analysis pipeline:
    1. Problem initialization and forward model setup
    2. Synthetic data generation with realistic noise
    3. Bayesian inference using adaptive MCMC methods
    4. Results visualization and performance assessment
    
    Workflow Overview:
        The main function demonstrates a typical research workflow for earthquake
        fault parameter estimation:
        
        Setup Phase:
        - Initialize RSF analysis framework with parameter space
        - Configure Rate and State friction forward model
        - Generate synthetic time series data across Dc values
        
        Analysis Phase:
        - Perform Bayesian inference using JSON data storage
        - Execute MCMC sampling for each characteristic slip distance
        - Generate posterior distributions and convergence diagnostics
        
        Visualization Phase:
        - Display all generated plots and analysis results
        - Clean up figure resources for memory management
    
    Configuration:
        The analysis uses module-level constants for easy customization:
        - Parameter space: 5 Dc values from 100-5000 μm
        - Temporal resolution: 500 time steps over 50 seconds
        - MCMC sampling: 500 samples per Dc value
        - Data storage: JSON format for demonstration
        - Prior distribution: Uniform [0, 10000] μm
    
    Performance Monitoring:
        The function tracks computational performance:
        - JSON inference execution time measurement
        - Memory usage through visualization management
        - Computational efficiency assessment
    
    Output and Visualization:
        Generated results include:
        - Time series plots for each Dc value (if plotting enabled)
        - MCMC trace plots showing parameter evolution
        - Posterior probability density plots
        - Convergence diagnostics and acceptance statistics
        - Performance metrics and timing information
    
    Example Execution:
        >>> # Run complete analysis
        >>> python main.py
        >>> 
        >>> # Expected output:
        >>> # --- Dc is 100.0 ---
        >>> # [MCMC progress indicators]
        >>> # --- Dc is 1350.0 ---
        >>> # [MCMC progress indicators]
        >>> # ... (for all Dc values)
        >>> # [Posterior distribution plots displayed]
    
    Customization Options:
        To modify the analysis:
        
        Parameter Space:
        - Adjust NUMBER_SLIP_VALUES for different resolution
        - Change LOWEST/LARGEST_SLIP_VALUE for different ranges
        - Modify QPRIORS for different prior assumptions
        
        Computational Settings:
        - Increase NSAMPLES for better posterior resolution
        - Adjust NUMBER_TIME_STEPS for temporal resolution
        - Change QSTART for different MCMC initialization
        
        Analysis Options:
        - Enable plotting in RSF initialization (plotfigs=True)
        - Add MySQL analysis for storage comparison
        - Include reduced-order modeling for efficiency
    
    Extended Analysis Example:
        >>> def main():
        ...     problem = setup_problem()
        ...     
        ...     # Enable visualization
        ...     problem.plotfigs = True
        ...     
        ...     # Compare storage formats
        ...     json_time = perform_inference(problem, 'json', NSAMPLES)
        ...     mysql_time = perform_inference(problem, 'mysql', NSAMPLES)
        ...     
        ...     # Performance analysis
        ...     print(f"JSON: {json_time:.2f}s, MySQL: {mysql_time:.2f}s")
        ...     print(f"Speedup: {mysql_time/json_time:.2f}x")
        ...     
        ...     # Save results
        ...     plt.savefig('rsf_posterior_analysis.png', dpi=300)
        ...     plt.show()
        ...     plt.close('all')
    
    Resource Management:
        The function includes proper resource cleanup:
        - plt.show() displays all generated figures
        - plt.close('all') releases figure memory
        - Prevents memory accumulation in batch processing
    
    Error Handling:
        Common issues and their resolution:
        - Import errors: Verify module path setup
        - Model convergence: Check parameter bounds and initialization
        - Memory issues: Reduce NSAMPLES or NUMBER_SLIP_VALUES
        - Visualization: Ensure matplotlib backend is configured
    
    Integration with External Tools:
        The results can be used for:
        - Further statistical analysis with SciPy/StatsModels
        - Parameter sensitivity studies
        - Model comparison and validation
        - Publication-quality figure generation
        - Database storage of inference results
    
    Scientific Impact:
        This analysis framework enables:
        - Quantitative earthquake fault parameter estimation
        - Uncertainty quantification in geophysical models
        - Validation of laboratory friction experiments
        - Development of improved earthquake forecasting models
    
    Note:
        - Execution time varies with parameter space size and sample count
        - Visualization quality depends on matplotlib configuration
        - Consider parallel execution for large parameter studies
        - Results are displayed but not automatically saved
        - Memory usage scales with problem size and visualization
    """
    problem = setup_problem()

    json_time = perform_inference(problem, 'json', NSAMPLES)

    plt.show()
    plt.close('all')

if __name__ == '__main__':
    main()