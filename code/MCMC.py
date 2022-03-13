from matplotlib.pyplot import flag
import numpy as np
import copy
from scipy.stats import gamma 
import sys
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class MCMC:
    """
    Class for MCMC sampling
    """
    def __init__(self,model,qpriors,nsamples,nburn,data,problem_type,lstm_model,qstart=None,adapt_interval=100,verbose=True):
        """
        Initialize the sampling process
        """
        self.model=model
        self.qstart=qstart
        self.qpriors=qpriors
        self.nsamples=nsamples
        self.nburn=nburn
        self.verbose=verbose
        self.adapt_interval=adapt_interval
        self.data=data
        self.lstm_model=lstm_model
        self.problem_type=problem_type
        self.consts=model.consts
        self.n0 = 0.01

        # initial guess!!
        for arg in self.qstart.keys(): 
            self.consts[arg]=qstart[arg] 
        
        if(self.problem_type=='full'): 
        # high fidelity model
           t_,acc_,temp_=self.model.evaluate(self.consts)
        else: 
        # reduced order model
           t_,acc_=self.model.rom_evaluate(self.consts,lstm_model)

        #constructing initial covariance matrix !!!
        acc = acc_[:,0] 
        acc = acc.reshape(1,acc.shape[0])
        self.std2=[np.sum((acc-self.data)**2,axis=1)[0]/(acc.shape[1]-len(self.qpriors.keys()))]
        X=[]

        for arg in self.qpriors.keys(): 

            consts_dq=copy.deepcopy(self.consts) 
            consts_dq[arg]=consts_dq[arg]*(1+1e-6) 

            if(self.problem_type=='full'): # high fidelity model
               t_,acc_dq_,temp_=self.model.evaluate(consts_dq)
            else: # reduced order model
               t_,acc_dq_=self.model.rom_evaluate(consts_dq,lstm_model)

            acc_dq = acc_dq_[:,0] 
            acc_dq = acc_dq.reshape(1,acc_dq.shape[0])
            print(acc[0,:])
            print('---------')
            print(acc_dq[0,:])
            X.append((acc_dq[0,:]-acc[0,:])/(consts_dq[arg]*1e-6)) 

        # Chapter 8 in Ralf Smith's book !!!
        X=np.asarray(X).T
        X=np.linalg.inv(np.dot(X.T,X))
        self.Vstart=self.std2[0]*X 

        self.qstart_vect=np.zeros((len(self.qstart),1))
        self.qstart_limits=np.zeros((len(self.qstart),2))
        flag=0
        for arg in self.qstart.keys(): 

            self.qstart_vect[flag,0]=self.qstart[arg] 
            self.qstart_limits[flag,0]=self.qpriors[arg][1] 
            self.qstart_limits[flag,1]=self.qpriors[arg][2] 
            flag=flag+1


    def sample(self):
        """
        Function for sampling using adaptive Metropolis algorithm
        Return:
            Q_MCMC: Accepted samples
        """
        qparams=copy.deepcopy(self.qstart_vect) 
        qmean_old=copy.deepcopy(self.qstart_vect) 
        qmean=copy.deepcopy(self.qstart_vect) 

        Vold=copy.deepcopy(self.Vstart)  
        Vnew=copy.deepcopy(self.Vstart) 

        SSqprev=self.SSqcalc(qparams) # squared error
        iaccept=0

        for isample in range(0,self.nsamples):
 
            #q_new is randomly sampled from a normal distribution with mean being the last element of qparams!!!
            q_new = np.reshape(np.random.multivariate_normal(qparams[:,-1],Vold),(-1,1)) 

            accept,SSqnew=self.acceptreject(q_new,SSqprev,self.std2[-1])
            print(isample,accept)
            print("Generated sample ---- ",np.asscalar(q_new))

            if accept:
                qparams=np.concatenate((qparams,q_new),axis=1)
                SSqprev=copy.deepcopy(SSqnew)
                iaccept=iaccept+1
            else:
                q_new=np.reshape(qparams[:,-1],(-1,1))
                qparams=np.concatenate((qparams,q_new),axis=1)

            # Update standard deviation
            # Chapter 8 in Ralf Smith's book !!!
            aval=0.5*(self.n0+self.data.shape[1]);
            bval=0.5*(self.n0*self.std2[-1]+SSqprev);
            self.std2.append(1/gamma.rvs(aval,scale=1/bval,size=1)[0])

            if np.mod((isample+1),self.adapt_interval)==0:
                try:
                    Vnew=2.38**2/len(self.qpriors.keys())*np.cov(qparams[:,-self.adapt_interval:])
                    if qparams.shape[0]==1:
                        Vnew=np.reshape(Vnew,(-1,1))
                    R = np.linalg.cholesky(Vnew)
                    Vold=copy.deepcopy(Vnew)
                except:
                    pass
        
        print("acceptance ratio:",iaccept/self.nsamples)
        self.std2=np.asarray(self.std2)[self.nburn:]            
        return qparams[:,self.nburn:]


    def acceptreject(self,q_new,SSqprev,std2):
   
        accept=np.all(q_new[:,0]>self.qstart_limits[:,0]) and np.all(q_new[:,0]<self.qstart_limits[:,1])
       
        if accept:
            SSqnew=self.SSqcalc(q_new)
            #if r is greater than a number randomly sampled between 0 and 1  
            accept=(min(0,0.5*(SSqprev-SSqnew)/std2)>np.log(np.random.rand(1))[0]) 

        if accept:
            return accept,SSqnew
        else:
            return accept,SSqprev

            
    def SSqcalc(self,q_new):

        flag=0
        for arg in self.qstart.keys(): 
            consts_dq=copy.deepcopy(self.consts)
            consts_dq[arg]=q_new[flag,]
            flag=flag+1

        if(self.problem_type=='full'): 
        # high fidelity model
           t_,acc_,temp_=self.model.evaluate(consts_dq)
        else: 
        # reduced order model
           t_,acc_=self.model.rom_evaluate(consts_dq,self.lstm_model)

        acc = acc_[:,0] 
        acc = acc.reshape(1,acc.shape[0])
        SSq=np.sum((acc-self.data)**2,axis=1) # squared error

        return SSq


    def plot_dist(self, qparams, plot_title, dc):

        n_rows = 1
        n_columns = 2
        gridspec = {'width_ratios': [0.7, 0.15], 'wspace': 0.15}
        fig, ax = plt.subplots(n_rows, n_columns, gridspec_kw=gridspec)
#        fig.suptitle('$d_c=%s\,\mu m$ for %s model' % (dc,plot_title), fontsize=14)
        fig.suptitle('$d_c=%s\,\mu m$' % dc, fontsize=14)
        ax[0].plot(qparams[0,:], 'b-', linewidth=1.0)
        ylims = ax[0].get_ylim()
        x = np.linspace(ylims[0], ylims[1], 1000)
        kde = gaussian_kde(qparams[0,:])
        ax[1].plot(kde.pdf(x), x, 'b-')
        max_val = x[kde.pdf(x).argmax()]
        ax[1].plot(kde.pdf(x)[kde.pdf(x).argmax()],max_val, 'ro')
        ax[1].annotate(str(round(max_val,2)),xy=(0.90*kde.pdf(x)[kde.pdf(x).argmax()],max_val),size=14)
        ax[1].fill_betweenx(x, kde.pdf(x), np.zeros(x.shape), alpha=0.3)
        ax[1].set_xlim(0, None)
        ax[0].set_ylabel('$d_c$', fontsize=14)
        ax[0].set_xlim(0, qparams.shape[1]) 
        ax[0].set_xlabel('Sample number')
        ax[1].set_xlabel('Prob. density')
        ax[1].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(True)
        ax[1].get_xaxis().set_ticks([])
        fig.savefig('./plots/burn_%s_%s_%s_.png' % (plot_title,dc,sys.argv[2]))
