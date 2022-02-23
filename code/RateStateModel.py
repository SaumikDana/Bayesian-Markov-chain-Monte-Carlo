from scipy.misc import derivative
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from math import exp,log,pi,sin,cos
import torch

class RateStateModel:
    """Class for rate and state model"""

    def __init__(self,t_start,t_end,num_tsteps,window,plotfigs=False,plotname="test.png"):

        consts={}
        consts["a"]=0.011;  consts["b"]=0.014;   consts["mu_ref"]=0.6
        consts["V_ref"]= 1;   consts["k1"]= 1e-7

        # Time range
        consts["t_start"]=t_start;  consts["t_final"]=t_end; 
        consts["delta_t"]=(t_end-t_start)/num_tsteps

        # Initial conditions
        consts["mu_t_zero"] = 0.6;  consts["V_ref"] = 1.0
        
        consts["mu_ref"] = 0.6
        consts["RadiationDamping"]=True  
        consts["window"]=window 
 
        self.consts=consts
        self.plotfigs=plotfigs
        self.plotname=plotname

    def set_dc(self,dc):

        self.consts["Dc"] = dc

    def friction(self,t,y):

        # Just to help readability
        #y[0] is mu (friction)
        #y[1] is theta
        #y[2] is velocity

        # effective spring stiffness
        kprime = 1e-2*10/self.consts["Dc"]

        # loading
        a1 = 20
        a2 = 10
        V_l = self.consts["V_ref"]*(1+exp(-t/a1)*sin(a2*t))

        n = len(y)
        dydt = np.zeros((n,1))

        # compute v
        temp_ = self.consts["V_ref"] * y[1] / self.consts["Dc"]
        temp = 1/self.consts["a"]*(y[0] - self.consts["mu_ref"] - self.consts["b"] * log(temp_))
        v = self.consts["V_ref"] * exp(temp)

        # time derivative of theta
        dydt[1] = 1. - v * y[1] / self.consts["Dc"]

        # double derivative of theta
        ddtheta = - dydt[1]*v/ self.consts["Dc"]

        # time derivative of mu
        dydt[0] = kprime*V_l - kprime*v

        # time derivative of velocity
        dydt[2] = v/self.consts["a"]*(dydt[0] - self.consts["b"]/y[1]*dydt[1])

        if self.consts["RadiationDamping"]:
            # radiation damping term
            dydt[0] = dydt[0] - self.consts["k1"]*dydt[2]
            dydt[2] = v/self.consts["a"]*(dydt[0] - self.consts["b"]/y[1]*dydt[1])

        return dydt

    def evaluate(self,params):

        for arg in params.keys():
            self.consts[arg]=params[arg]
        
        r = integrate.ode(self.friction).set_integrator('vode', order=5,max_step=0.001,method='bdf',atol=1e-10,rtol=1e-6)
        num_steps = int(np.floor((self.consts["t_final"]-self.consts["t_start"])/self.consts["delta_t"]))

        theta_t_zero = self.consts["Dc"]/self.consts["V_ref"]
        v = self.consts["V_ref"]
        r.set_initial_value([self.consts["mu_t_zero"], theta_t_zero, self.consts["V_ref"]], self.consts["t_start"])

        # Create arrays to store trajectory
        t,mu,theta,velocity,acc = np.zeros((num_steps,1)),np.zeros((num_steps,1)),np.zeros((num_steps,1)),np.zeros((num_steps,1)),np.zeros((num_steps,1))
        t[0] = self.consts["t_start"]
        mu[0] = self.consts["mu_ref"]
        theta[0] = theta_t_zero
        velocity[0] = v
        acc[0] = 0

        # Integrate the ODE(s) across each delta_t timestep
        k = 1
        while r.successful() and k < num_steps:
            #integrate.ode.set_f_params(r,velocity,k)
            r.integrate(r.t + self.consts["delta_t"])
            # Store the results to plot later
            t[k] = r.t
            mu[k] = r.y[0]
            theta[k] = r.y[1]
            velocity[k] = r.y[2]
            acc[k] = (velocity[k]-velocity[k-1])/self.consts["delta_t"]
            k += 1

        acc_noise = acc+1.0*np.abs(acc)*np.random.randn(acc.shape[0],acc.shape[1]) #synthetic data

        t_,acc_,acc_noise_ = np.zeros((num_steps,2)),np.zeros((num_steps,2)),np.zeros((num_steps,2))
        t_[:,0] = t[:,0]
        t_[:,1] = self.consts["Dc"]

        acc_noise_[:,0] = acc_noise[:,0]
        acc_noise_[:,1] = acc_noise[:,0]

        acc_[:,0] = acc[:,0]
        acc_[:,1] = acc[:,0]

        del(t,mu,theta,velocity,acc,acc_noise)
 
        if self.plotfigs:
            self.generateplots(t,mu,theta,velocity,acc)
            self.plotfigs = False # only once!!!
        # Make some plots
        return t_,acc_,acc_noise_


    def rom_evaluate(self,params,lstm_model):
         
        for arg in params.keys():
            self.consts[arg]=params[arg]

        num_steps = int(np.floor((self.consts["t_final"]-self.consts["t_start"])/self.consts["delta_t"]))
        window = self.consts["window"]
    
        # Create arrays to store trajectory
        t = np.zeros((num_steps,2)) 
        acc = np.zeros((num_steps,2)) 
    
        t[0,0] = self.consts["t_start"]
    
        k = 1
        while k < num_steps:
            t[k,0] = t[k-1,0] + self.consts["delta_t"]
            k += 1
    
        t[:,1] = self.consts["Dc"]
        num_steps_ = int(num_steps/window) 
        train_plt = np.zeros((window,2)) 
     
        for ii in range(num_steps_):
    
            start = ii*window
            end = start + window
    
            train_plt[0:window,0] = t[start:end,0]
            train_plt[0:window,1] = t[start:end,1]
            Y_train_pred = lstm_model.predict(torch.from_numpy(train_plt).type(torch.Tensor), target_len = window)
            acc[start:end,0] = Y_train_pred[:, 0]
            acc[start:end,1] = Y_train_pred[:, 0]
        
        return t, acc

   
    def generateplots(self,t,mu,theta,velocity,acc):
 
        plt.figure()
        plt.title('$d_c$='+str(self.consts["Dc"])+' $\mu m$' + ' RSF solution')
        plt.plot(t, acc, 'b', linewidth=1.0)
        plt.xlim(self.consts["t_start"]-2.0, self.consts["t_final"])
        plt.xlabel('Time (sec)')
        plt.ylabel('Acceleration $(\mu m/s^2)$')
        plt.grid('on')
        plt.savefig("./plots/smooth.png")

        plt.figure()
        plt.title('$d_c$='+str(self.consts["Dc"])+' $\mu m$' + ' Noise added')
        acc_noise=acc+1.0*np.abs(acc)*np.random.randn(acc.shape[0],acc.shape[1]) #synthetic data
        plt.plot(t, acc_noise, 'b', linewidth=1.0)
        plt.xlim(self.consts["t_start"]-2.0, self.consts["t_final"])
        plt.xlabel('Time (sec)')
        plt.ylabel('Acceleration $(\mu m/s^2)$')
        plt.grid('on')
        plt.savefig("./plots/noisy_1.png")