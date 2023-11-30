from scipy.misc import derivative
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from math import exp, log, pi, sin, cos
from lstm.utils import RateStateModel as RateStateModel_base

# Constants
A = 0.011
B = 0.014
MU_REF = 0.6
V_REF = 1.
K1 = 1.E-7
START_TIME = 0.0
END_TIME = 50.0

class RateStateModel(RateStateModel_base):
    """ 
    Class for rate and state model 
    """

    def __init__(
        self, 
        number_time_steps=500, 
        start_time=START_TIME, 
        end_time=END_TIME):

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
        
        def friction(t, y):

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
