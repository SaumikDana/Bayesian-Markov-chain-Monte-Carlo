from RateStateModel import RateStateModel
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    consts={}
    consts["k"]=1e-1;   consts["a"]=0.011;  consts["b"]=0.014;   consts["mu_ref"]=0.6
    consts["V_ref"]= 1;   consts["k1"]= 1e-7

    # Time range
    consts["t_start"]=0.0;  consts["t_final"]=50.0; consts["delta_t"]=1e-2

    # Initial conditions
    consts["mu_t_zero"] = 0.6;  consts["V_ref"] = 1.0;  consts["Dc"] = 1.0; consts["mu_ref"] = 0.6

    params={"Dc":1.0}

    consts["RadiationDamping"]=False    
    model=RateStateModel(consts,plotfigs=False,plotname="no_radiation_damping_"+str(params["Dc"])+".png")
    acc=model.evaluate(params)

    acc_noise=acc+5.0*np.abs(acc)*np.random.randn(acc.shape[0],acc.shape[1])

    fig=plt.figure()
    fig.suptitle('$d_c=1.0\,\mu m$')
#    plt.plot(acc_noise.T,'o', markersize=2.0)
    plt.plot(acc_noise.T-acc.T,linewidth=1.5)
#    plt.legend(["Synthetic data (has noise)","True data"])
    plt.xlabel("Time (s)")
    plt.ylabel("Noise in Acceleration ($\mu m/s^2$)")
    # plt.xlim([0,500])
    fig.savefig("../outputs/acceleration_data1aa.png",bbox_inches='tight')
    plt.show()


    
