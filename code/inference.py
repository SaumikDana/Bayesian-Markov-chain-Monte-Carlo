from MCMC import MCMC
import sys
from save_load import save_object, load_object

def inference(file1,file2,num_dc,num_tsteps,dc_,model):
    # load objects!!!
    model_lstm = load_object(file1)  # ROM
    acc_appended_noise = load_object(file2) # noisy data

    for ii in range(0,num_dc):

        if ii == 2 or ii == 11 or ii == 13 or ii == 15 or ii == 16 or ii == 17:
           # noisy data!!!
           acc = acc_appended_noise[ii*num_tsteps:ii*num_tsteps+num_tsteps,0]
           acc = acc.reshape(1, num_tsteps)
     
           dc = dc_[ii]
           print('--- dc is %s ---' % dc)

           qstart={"Dc":100} # initial guess
           qpriors={"Dc":["Uniform",0.1, 1000]}

           nsamples = int(sys.argv[2])
           nburn = nsamples/2
           
           problem_type = 'full'
           MCMCobj1=MCMC(model,qpriors=qpriors,nsamples=nsamples,nburn=nburn,data=acc,problem_type=problem_type,lstm_model=model_lstm,qstart=qstart,adapt_interval=10,verbose=True)
           qparams1=MCMCobj1.sample() # run the Bayesian/MCMC algorithm
           std_MCMC1 = MCMCobj1.std2
           MCMCobj1.plot_dist(qparams1,'full',dc)

           problem_type = 'rom'
           MCMCobj2=MCMC(model,qpriors=qpriors,nsamples=nsamples,nburn=nburn,data=acc,problem_type=problem_type,lstm_model=model_lstm,qstart=qstart,adapt_interval=10,verbose=True)
           qparams2=MCMCobj2.sample() # run the Bayesian/MCMC algorithm
           std_MCMC2 = MCMCobj2.std2
           MCMCobj2.plot_dist(qparams2,'reduced order',dc)


