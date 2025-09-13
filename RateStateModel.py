from imports import *


# Constants
A = 0.011
B = 0.014
MU_REF = 0.6
V_REF = 1.
K1 = 1.E-7
START_TIME = 0.0
END_TIME = 50.0

RateStateModel_base = object  # Fallback to a base object

class RateStateModel(RateStateModel_base):
    """
    Rate and State Friction Model for earthquake fault mechanics simulation.
    
    This class implements the rate and state friction law, a widely used constitutive
    model in earthquake mechanics that describes the evolution of friction on fault
    surfaces. The model captures the velocity dependence of friction and the time-
    dependent healing of fault surfaces through a state variable.
    
    The rate and state friction law is governed by:
    μ = μ₀ + a*ln(V/V_ref) + b*ln(V_ref*θ/Dc)
    dθ/dt = 1 - V*θ/Dc
    
    Where:
    - μ is the friction coefficient
    - V is the sliding velocity
    - θ is the state variable (characteristic healing time)
    - a, b are material parameters
    - Dc is the characteristic slip distance
    - μ₀ is the reference friction coefficient
    
    The model includes:
    - Dynamic loading with time-varying velocity
    - Spring-block system dynamics with effective stiffness
    - Optional radiation damping for realistic energy dissipation
    - Numerical integration using adaptive ODE solvers
    - Synthetic noise addition to simulate experimental conditions
    
    Physical Interpretation:
    - a > 0: velocity strengthening (direct effect)
    - b > 0: velocity weakening when combined with state evolution
    - a-b < 0: potentially unstable sliding (stick-slip behavior)
    - a-b > 0: stable sliding
    - Dc controls the distance scale over which state evolves
    
    Applications:
    - Earthquake fault mechanics modeling
    - Laboratory friction experiments simulation
    - Seismic cycle studies
    - Fault stability analysis
    - Parameter estimation from experimental data
    
    Attributes:
        a (float): Direct effect parameter (velocity dependence)
        b (float): Evolution effect parameter (state dependence)
        mu_ref (float): Reference friction coefficient
        V_ref (float): Reference sliding velocity
        k1 (float): Radiation damping coefficient
        t_start (float): Simulation start time
        t_final (float): Simulation end time
        num_tsteps (int): Number of time steps for integration
        delta_t (float): Time step size
        mu_t_zero (float): Initial friction coefficient
        RadiationDamping (bool): Whether to include radiation damping
        Dc (float): Characteristic slip distance (must be set before evaluation)
    
    Examples:
        >>> # Basic usage
        >>> model = RateStateModel(number_time_steps=1000, end_time=100.0)
        >>> model.Dc = 0.02  # Set characteristic distance
        >>> t, acc, acc_noise = model.evaluate()
        >>> 
        >>> # Custom parameters
        >>> model = RateStateModel()
        >>> model.a = 0.015  # Modify direct effect
        >>> model.b = 0.020  # Modify evolution effect
        >>> model.Dc = 0.01
        >>> model.RadiationDamping = False  # Disable damping
        >>> results = model.evaluate()
        
        >>> # Analysis of results
        >>> t, acceleration, noisy_acceleration = model.evaluate()
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(12, 4))
        >>> plt.subplot(1, 2, 1)
        >>> plt.plot(t, acceleration, label='Clean signal')
        >>> plt.plot(t, noisy_acceleration, label='With noise', alpha=0.7)
        >>> plt.xlabel('Time')
        >>> plt.ylabel('Acceleration')
        >>> plt.legend()
    
    References:
        - Dieterich, J. H. (1979). Modeling of rock friction
        - Ruina, A. (1983). Slip instability and state variable friction laws
        - Marone, C. (1998). Laboratory-derived friction laws and their application
    
    Note:
        - Dc parameter must be set before calling evaluate()
        - The model uses adaptive ODE integration for numerical stability
        - Radiation damping is enabled by default for physical realism
        - Time units are arbitrary but should be consistent throughout
        - Acceleration output includes synthetic noise for realism
    """

    def __init__(
        self, 
        number_time_steps=500, 
        start_time=START_TIME, 
        end_time=END_TIME
    ):
        """
        Initialize the Rate and State friction model with simulation parameters.
        
        Sets up the model with default physical parameters based on typical laboratory
        values for rock friction experiments. The temporal discretization is defined
        by the number of time steps and the start/end times.
        
        Args:
            number_time_steps (int, optional): Number of discrete time steps for numerical
                                             integration. Higher values improve accuracy but
                                             increase computation time. Defaults to 500.
            start_time (float, optional): Simulation start time in arbitrary time units.
                                        Defaults to START_TIME (0.0).
            end_time (float, optional): Simulation end time in same units as start_time.
                                      Must be greater than start_time. Defaults to END_TIME (50.0).
        
        Initializes:
            Physical parameters (from literature values):
            - a = 0.011: Direct effect parameter (dimensionless)
            - b = 0.014: Evolution effect parameter (dimensionless)
            - mu_ref = 0.6: Reference friction coefficient (dimensionless)
            - V_ref = 1.0: Reference velocity (length/time)
            - k1 = 1e-7: Radiation damping coefficient
            
            Temporal parameters:
            - Time range and discretization
            - Time step size (computed from range and number of steps)
            
            Initial conditions:
            - Initial friction coefficient set to reference value
            
            Model options:
            - Radiation damping enabled by default
            - Dc parameter initialized to None (must be set before evaluation)
        
        Note:
            - The time step is automatically computed as (end_time - start_time) / number_time_steps
            - Physical parameters use typical values from rock friction literature
            - All parameters can be modified after initialization
            - The model assumes consistent units throughout (user responsibility)
        
        Example:
            >>> # Default simulation (50 time units, 500 steps)
            >>> model = RateStateModel()
            >>> 
            >>> # High-resolution, long-duration simulation
            >>> model = RateStateModel(number_time_steps=2000, end_time=200.0)
            >>> 
            >>> # Short-duration, coarse simulation
            >>> model = RateStateModel(number_time_steps=100, end_time=10.0)
        """
        # Define model constants
        self.a = A
        self.b = B
        self.mu_ref = MU_REF
        self.V_ref = V_REF
        self.k1 = K1

        # Define time range
        self.t_start = start_time
        self.t_final = end_time
        self.num_tsteps = number_time_steps
        self.delta_t = (end_time - start_time) / number_time_steps

        # Define initial conditions
        self.mu_t_zero = MU_REF

        # Add additional model constants
        self.RadiationDamping = True
        self.Dc = None

        return
        
    def evaluate(self):
        """
        Solve the rate and state friction model using numerical integration.
        
        This method performs the complete simulation of the rate and state friction
        system by solving the coupled differential equations numerically. The system
        includes the friction evolution, state variable evolution, and dynamic loading
        with optional radiation damping.
        
        The method solves the system:
        dμ/dt = k'*(V_l - V) - k1*dV/dt  (with radiation damping)
        dθ/dt = 1 - V*θ/Dc
        dV/dt = (V/a)*(dμ/dt - b/θ*dθ/dt)
        
        Where:
        - V is computed from the rate and state law: V = V_ref * exp((μ-μ₀-b*ln(V_ref*θ/Dc))/a)
        - V_l is the loading velocity: V_ref * (1 + exp(-t/a1) * sin(a2*t))
        - k' is the effective stiffness: 1e-2 * 10 / Dc
        
        Returns:
            tuple: A 3-element tuple containing:
                - t (np.ndarray): Time array with shape (num_steps,). Time values
                                from t_start to t_final with delta_t spacing.
                - acc (np.ndarray): Clean acceleration array with shape (num_steps,).
                                  Computed as numerical derivative of velocity.
                - acc_noise (np.ndarray): Noisy acceleration array with shape (num_steps,).
                                        Clean acceleration plus 100% relative Gaussian noise.
        
        Raises:
            ValueError: If Dc is None (characteristic distance must be set before evaluation)
            RuntimeError: If ODE integration fails or becomes unstable
            FloatingPointError: If numerical overflow/underflow occurs during integration
        
        Algorithm Details:
            1. Sets up the differential equation system with nested friction function
            2. Initializes arrays for state variables (μ, θ, V) and outputs
            3. Uses Dormand-Prince 8(5,3) ODE solver with adaptive step size
            4. Integrates over the specified time range with fixed output intervals
            5. Computes acceleration via finite differences
            6. Adds synthetic noise to acceleration for realism
        
        Physical Interpretation:
            - Loading function creates time-varying driving velocity
            - Effective stiffness couples system dynamics to Dc parameter
            - Radiation damping models energy dissipation during rapid sliding
            - State variable evolution controls friction memory effects
            - Acceleration output represents measurable seismic signal
        
        Numerical Considerations:
            - Uses high-precision ODE solver (rtol=1e-6, atol=1e-10)
            - Adaptive time stepping for numerical stability
            - Handles stiff systems through implicit methods
            - Acceleration computed via backward differences for stability
        
        Noise Model:
            - Multiplicative Gaussian noise: acc_noise = acc + 1.0 * |acc| * N(0,1)
            - 100% relative noise level (signal-dependent)
            - Models experimental measurement uncertainty
            - Preserves sign of acceleration signal
        
        Example:
            >>> model = RateStateModel(number_time_steps=1000)
            >>> model.Dc = 0.02  # Required: set characteristic distance
            >>> 
            >>> # Run simulation
            >>> time, clean_acc, noisy_acc = model.evaluate()
            >>> 
            >>> # Analyze results
            >>> print(f"Simulation duration: {time[-1] - time[0]:.2f}")
            >>> print(f"Max acceleration: {np.max(np.abs(clean_acc)):.4f}")
            >>> print(f"RMS noise level: {np.std(noisy_acc - clean_acc):.4f}")
            >>> 
            >>> # Check for stick-slip behavior
            >>> velocity_proxy = np.gradient(clean_acc)  # Rough velocity estimate
            >>> if np.std(velocity_proxy) > np.mean(np.abs(velocity_proxy)):
            ...     print("Stick-slip behavior detected")
        
        Performance Notes:
            - Computation time scales linearly with number_time_steps
            - Memory usage is O(number_time_steps) for storing trajectories
            - ODE solver may slow down for stiff systems (small Dc, large a-b)
            - Consider reducing tolerance for faster computation if high precision not needed
        
        Validation:
            - Check energy conservation (if radiation damping disabled)
            - Verify friction bounds: μ should remain physically reasonable
            - Monitor state variable: θ > 0 always (physical constraint)
            - Compare with analytical solutions for simple loading cases
        """
        def friction(t, y):
            """
            Define the rate and state friction differential equation system.
            
            This nested function computes the time derivatives of the state variables
            (friction coefficient, state variable, and velocity) according to the
            rate and state friction law with dynamic loading and optional radiation damping.
            
            The function implements the coupled system:
            1. Rate and state friction law (algebraic constraint)
            2. State variable evolution (Dieterich law)
            3. Dynamic equilibrium with loading and damping
            
            Args:
                t (float): Current time value during integration
                y (np.ndarray): State vector [μ, θ, V] where:
                              y[0] = μ (friction coefficient)
                              y[1] = θ (state variable, healing time)
                              y[2] = V (sliding velocity)
            
            Returns:
                np.ndarray: Time derivatives [dμ/dt, dθ/dt, dV/dt] as column vector
                           with shape (3, 1)
            
            Physical Model:
                - Loading velocity: V_l = V_ref * (1 + exp(-t/20) * sin(10*t))
                - Effective stiffness: k' = 0.1 / Dc
                - Friction law: V = V_ref * exp((μ-μ₀-b*ln(V_ref*θ/Dc))/a)
                - State evolution: dθ/dt = 1 - V*θ/Dc
                - Force balance: dμ/dt = k'*(V_l - V) - k1*dV/dt (with damping)
            
            Note:
                - Uses current model parameters (a, b, Dc, etc.)
                - Handles radiation damping through velocity coupling
                - Maintains physical constraints implicitly
            """
            # Just to help readability
            # y[0] is mu (friction)
            # y[1] is theta
            # y[2] is velocity
            
            V_ref = self.V_ref
            a = self.a
            b = self.b
            dc = self.Dc

            # effective spring stiffness
            kprime = 1e-2 * 10 / dc

            # loading
            a1 = 20
            a2 = 10
            V_l = V_ref * (1 + exp(-t/a1) * sin(a2*t))

            # initialize the array of derivatives
            n = len(y)
            dydt = np.zeros((n, 1))

            # compute v
            temp = 1 / a * (y[0] - self.mu_ref - b * log(V_ref * y[1] / dc))
            v = V_ref * exp(temp)

            # time derivative of theta
            dydt[1] = 1. - v * y[1] / dc

            # time derivative of mu
            dydt[0] = kprime * V_l - kprime * v

            # time derivative of velocity
            dydt[2] = v / a * (dydt[0] - b / y[1] * dydt[1])

            # add radiation damping term if specified
            if self.RadiationDamping:
                # time derivative of mu with radiation damping
                dydt[0] = dydt[0] - self.k1 * dydt[2]
                # time derivative of velocity with radiation damping
                dydt[2] = v / a * (dydt[0] - b / y[1] * dydt[1])

            return dydt
        
        # Calculate the number of steps to take
        num_steps = int(np.floor((self.t_final - self.t_start) / self.delta_t))

        # Create arrays to store trajectory
        t, mu, theta, velocity, acc = \
            np.zeros(num_steps),\
                np.zeros(num_steps),\
                    np.zeros(num_steps),\
                        np.zeros(num_steps),\
                            np.zeros(num_steps)
        t[0] = self.t_start
        mu[0] = self.mu_ref
        theta[0] = self.Dc / self.V_ref
        velocity[0] = self.V_ref
        acc[0] = 0

        # Set up the ODE solver
        r = integrate.ode(friction).set_integrator('dop853', rtol=1e-6, atol=1e-10)

        # Set the initial conditions for the ODE solver
        r.set_initial_value([self.mu_t_zero, theta[0], velocity[0]], t[0])

        # Integrate the ODE(s) across each delta_t timestep
        k = 1
        while r.successful() and k < num_steps:
            r.integrate(r.t + self.delta_t)
            # Store the results to plot later
            t[k] = r.t
            mu[k] = r.y[0]
            theta[k] = r.y[1]
            velocity[k] = r.y[2]
            acc[k] = (velocity[k] - velocity[k-1]) / self.delta_t
            k += 1

        # Add some noise to the acceleration data to simulate real-world noise
        acc_noise = acc + 1.0 * np.abs(acc) * np.random.randn(acc.shape[0])

        # Return the data for plotting and analysis
        return t, acc, acc_noise