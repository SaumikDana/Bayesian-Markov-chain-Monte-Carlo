# use python3.7
from __future__ import division 
from scipy.misc import derivative
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from math import exp,log,pi,sin,cos
import random

fig = plt.figure()
fig.suptitle('d_c=10')

global Dc,V_ref,delta_t,t_start,t_final,amp,num_steps,k,c
Dc = 1000
V_ref = 1

# Trig-exp
t_start = 0.0
t_final = 50.
delta_t = 1e-1

num_steps = int((t_final-t_start)/delta_t)

def friction(t,y):
    a = 0.011
    b = 0.014
    kprime = 1e-2*10/Dc # inversely prop to Dc
    mu_ref = 0.6
    k1 = 1e-7 # radiation damping term

    a1 = 10
    a2 = 10
    V_l = V_ref*(1+exp(-t/a1)*sin(a2*t))

#    current_step = int((t-t_start)/delta_t)
#    V_l = V_ref*(1+x[current_step])

    # Just to help readability
    #y[0] is mu (friction)
    #y[1] is theta
    #y[2] is velocity

    n = len(y)
    dydt = np.zeros((n,1))

    # compute v
    temp_ = V_ref * y[1] / Dc
    temp = 1/a*(y[0] - mu_ref - b * log(temp_))
    v = V_ref * exp(temp)

    # time derivative of theta
    dydt[1] = 1. - v * y[1] / Dc

    # double derivative of theta
    ddtheta = - dydt[1]*v/ Dc

    # time derivative of mu
    dydt[0] = kprime*V_l - kprime*v

    # time derivative of velocity
    dydt[2] = v/a*(dydt[0] - b/y[1]*dydt[1])

    # radiation damping
    dydt[0] = dydt[0] - k1*dydt[2]
    dydt[2] = v/a*(dydt[0] - b/y[1]*dydt[1])
    
    return dydt

r = integrate.ode(friction).set_integrator('vode', order=5,max_step=0.001,method='bdf',atol=1e-10,rtol=1e-6)

# Initial conditions
mu_t_zero, mu_ref, theta_t_zero, v = 0.6, 0.6, Dc/V_ref, V_ref
r.set_initial_value([mu_t_zero, theta_t_zero, V_ref], t_start)

# Create arrays to store trajectory
t,mu,theta,velocity,acc = np.zeros((num_steps,1)),np.zeros((num_steps,1)),np.zeros((num_steps,1)),np.zeros((num_steps,1)),np.zeros((num_steps,1))
t[0],mu[0],theta[0],velocity[0],acc[0] = t_start,mu_ref,theta_t_zero,v,0

# Integrate the ODE(s) across each delta_t timestep
k = 1
while r.successful() and k < num_steps:
    #integrate.ode.set_f_params(r,velocity,k)
    r.integrate(r.t + delta_t)

    # Store the results to plot later
    t[k] = r.t
    mu[k] = r.y[0]
    theta[k] = r.y[1]
    velocity[k] = r.y[2]
    acc[k] = (velocity[k]-velocity[k-1])/delta_t
    k += 1

# Make some plots

ax4 = plt.subplot(111)
ax4.plot(t, acc, 'r', linewidth=0.25)
ax4.set_xlim(t_start, t_final)
ax4.set_xlabel('Time [sec]')
ax4.set_ylabel('Acceleration')
ax4.grid('on')

plt.show()

