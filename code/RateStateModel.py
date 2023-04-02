from scipy.misc import derivative
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from math import exp, log, pi, sin, cos
import torch

class RateStateModel:
    """Class for rate and state model"""

    def __init__(self, number_time_steps=500, start_time=0.0, end_time=50.0):

        # Define model constants
        self.a = 0.011
        self.b = 0.014
        self.mu_ref = 0.6
        self.V_ref = 1
        self.k1 = 1e-7

        # Define time range
        self.t_start = start_time
        self.t_final = end_time
        self.num_tsteps = number_time_steps
        self.delta_t = (end_time - start_time) / number_time_steps

        # Define initial conditions
        self.mu_t_zero = 0.6
        self.V_ref = 1.0

        # Add additional model constants
        self.RadiationDamping = True
        self.window = number_time_steps / 20
        self.Dc = None

        return

    def friction(self, t, y):

        # Just to help readability
        # y[0] is mu (friction)
        # y[1] is theta
        # y[2] is velocity

        # effective spring stiffness
        kprime = 1e-2 * 10 / self.Dc

        # loading
        a1 = 20
        a2 = 10
        V_l = self.V_ref * (1 + exp(-t/a1) * sin(a2*t))

        # initialize the array of derivatives
        n = len(y)
        dydt = np.zeros((n, 1))

        # compute v
        temp_ = self.V_ref * y[1] / self.Dc
        temp = 1 / self.a * (y[0] - self.mu_ref - self.b * log(temp_))
        v = self.V_ref * exp(temp)

        # time derivative of theta
        dydt[1] = 1. - v * y[1] / self.Dc

        # double derivative of theta
        ddtheta = - dydt[1] * v / self.Dc

        # time derivative of mu
        dydt[0] = kprime * V_l - kprime * v

        # time derivative of velocity
        dydt[2] = v / self.a * (dydt[0] - self.b / y[1] * dydt[1])

        # add radiation damping term if specified
        if self.RadiationDamping:
            # time derivative of mu with radiation damping
            dydt[0] = dydt[0] - self.k1 * dydt[2]
            # time derivative of velocity with radiation damping
            dydt[2] = v / self.a * (dydt[0] - self.b / y[1] * dydt[1])

        return dydt

    def evaluate(self, problem_type='full', lstm_model={}):
        # call either full model evaluation or reduced order model evaluation

        if problem_type == 'full':
            return self.full_evaluate()
        else:
            return self.rom_evaluate(lstm_model)
        
    def full_evaluate(self):
        
        # Set up the ODE solver
        r = integrate.ode(self.friction).set_integrator(
            'vode', order=5, max_step=0.001, method='bdf', atol=1e-10, rtol=1e-6)

        # Calculate the number of steps to take
        num_steps = int(np.floor((self.t_final - self.t_start) / self.delta_t))

        # Calculate the initial values for theta, mu, and velocity
        theta_t_zero = self.Dc / self.V_ref
        v = self.V_ref

        # Set the initial conditions for the ODE solver
        r.set_initial_value([self.mu_t_zero, theta_t_zero, self.V_ref], self.t_start)

        # Create arrays to store trajectory
        t, mu, theta, velocity, acc = np.zeros((num_steps, 1)), np.zeros((num_steps, 1)), np.zeros((num_steps, 1)), np.zeros((num_steps, 1)), np.zeros((num_steps, 1))
        t[0] = self.t_start
        mu[0] = self.mu_ref
        theta[0] = theta_t_zero
        velocity[0] = v
        acc[0] = 0

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
        acc_noise = acc + 1.0 * np.abs(acc) * np.random.randn(acc.shape[0], acc.shape[1])

        # Create arrays to store data for plotting
        t_, acc_, acc_noise_ = np.zeros((num_steps, 2)), np.zeros((num_steps, 2)), np.zeros((num_steps, 2))
        t_[:, 0] = t[:, 0]
        t_[:, 1] = self.Dc

        acc_noise_[:, 0] = acc_noise[:, 0]
        acc_noise_[:, 1] = acc_noise[:, 0]

        acc_[:, 0] = acc[:, 0]
        acc_[:, 1] = acc[:, 0]

        # Clean up
        del(t, mu, theta, velocity, acc, acc_noise)

        # Return the data for plotting and analysis
        return t_, acc_, acc_noise_

    def rom_evaluate(self,lstm_model):
         
        # Calculate the number of steps to take
        num_steps = int(np.floor((self.t_final - self.t_start) / self.delta_t))
        window = self.window

        # Create arrays to store trajectory
        t = np.zeros((num_steps, 2))
        acc = np.zeros((num_steps, 2))

        t[0, 0] = self.t_start

        # Calculate the time array
        k = 1
        while k < num_steps:
            t[k, 0] = t[k-1, 0] + self.delta_t
            k += 1

        t[:, 1] = self.Dc
        num_steps_ = int(num_steps / window)
        train_plt = np.zeros((window, 2))

        # Predict acceleration using the LSTM model
        for ii in range(num_steps_):
            start = ii * window
            end = start + window

            train_plt[0:window, 0] = t[start:end, 0]
            train_plt[0:window, 1] = t[start:end, 1]
            Y_train_pred = lstm_model.predict(
                torch.from_numpy(train_plt).type(torch.Tensor), target_len=window)
            acc[start:end, 0] = Y_train_pred[:, 0]
            acc[start:end, 1] = Y_train_pred[:, 0]

        # Return the time and acceleration arrays
        return t, acc
