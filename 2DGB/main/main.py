from algorithms import *
from factor_number import *
from simulation import *
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from scipy import stats
import scipy.linalg
import sympy
from sstudentt import SST


def factor_number_main(data,rmax,T=5):
    p2D2,q2D2 = ER2D2(data,rmax)
    pM,qM = ERPE(data,rmax,T=T)
    pH,qH = ERHuber(data,rmax,T=T)
    pop,qop = ER_initial(data,rmax)
    pF,qF = ER_iterate(data,rmax,T=T,r=0)
    return p2D2,q2D2,pM,qM,pH,qH,pop,qop,pF,qF

def estimation_comparison(estimation,size,RC): #estimation = [Rhat,Chat]
    p,q,p0,q0 = size
    R,C = RC
    Rhat,Chat = estimation
    PR = projection(R)
    PRhat = projection(Rhat)
    DRhat = Dmetric(PR,PRhat,p0)
    return DRhat

def algorithms_main(data,signal,size,RC,T=10):
    p,q,p0,q0 = size
    nd = len(data)
    R2D2,C2D2 = PCA2D2(data,p0,q0)
    DR2D2 = estimation_comparison([R2D2,C2D2],size,RC)
    RM,CM = PE(data,p0,q0,[R2D2,C2D2],T=T)
    DRM = estimation_comparison([RM,CM],size,RC)
    RH,CH,W = HuberPCA(data,p0,q0,[R2D2,C2D2],T=T)
    DRH = estimation_comparison([RH,CH],size,RC)
    Rop,Cop = GB_initial(data,p0,q0,[np.min([p0,q0])]*nd)
    DRop = estimation_comparison([Rop,Cop],size,RC) 
    RF,CF = GB_iterate(data,p0,q0,[np.min([p0,q0])]*nd,[Rop,Cop],T=T)
    DRF = estimation_comparison([RF,CF],size,RC)
    return DR2D2, DRM, DRH, DRop, DRF


def simulation_main(size,RC,N,dep,Fdist,Edist,sqrtSigma,sqrtOmega,Escale,rmax,sst=False,T=10): #size = [p,q,p0,q0], RC = [R,C], dep = [phi,psi], Edist = stats.t(1), sqrtOmega = [sqrtOmega1,sqrtOmega2]
    if sst == True:
        data,signal = simulation_sst(size,RC,N,dep,Fdist,Edist,sqrtSigma,sqrtOmega,Escale)
    else:
        data,signal = simulation(size,RC,N,dep,Fdist,Edist,sqrtSigma,sqrtOmega,Escale)
    p2D2,q2D2,pM,qM,pH,qH,pop,qop,pF,qF = factor_number_main(data,rmax,T=T)
    DR2D2, DRM, DRH, DRop, DRF = algorithms_main(data,signal,size,RC,T=T)
    return p2D2,q2D2,pM,qM,pH,qH,pop,qop,pF,qF, DR2D2, DRM, DRH, DRop, DRF

def main(size,N,dep,Fdist,Edist,sqrtSigma,sqrtOmega,Escale,rmax,sst=False,T=10,rep=100):
    Gauss = stats.norm()
    p,q,p0,q0 = size
    df = []
    for r in range(rep):
        R = Gauss.rvs(size=[p,p0])
        C = Gauss.rvs(size=[q,q0])
        row = simulation_main(size,[R,C],N,dep,Fdist,Edist,sqrtSigma,sqrtOmega,Escale,rmax,sst,T=T)
        df.append(row)
    df = pd.DataFrame(np.array(df),columns = ['p2D2','q2D2','pM','qM','pH','qH','pop','qop','pF','qF', 'DR2D2','DRM', 'DRH', 'DRop','DRF'])
    return df



if __name__ == '__main__':
    Gauss = stats.norm()
    Fdist = Gauss
    dict_Edist = {'Gauss':stats.norm(),'t(1)':stats.t(1),'skewt':SST(mu=0,sigma=np.sqrt(3),nu=2,tau=3),'t(3)':stats.t(3),'a-stable':stats.levy_stable(alpha=1.8,beta=0)}
    dep = [0.1,0.1]
    size_list = [[20,100,5,5],[100,20,5,5],[100,100,10,10]]
    n_list = [4]
    e_list = [2]

    for dist_name, Edist in dict_Edist.items():
        if dist_name == 'skewt':
            sst = True
        else:
            sst = False
        for size in size_list:
            p,q,p0,q0 = size
            rmax = p0*2
            sqrtSigma = [np.eye(p0),np.eye(q0)]
            sqrtOmega = [np.real(scipy.linalg.sqrtm(np.eye(p)*(1-1/p)+1*np.ones([p,p])/p)),np.real(scipy.linalg.sqrtm(np.eye(q)*(1-1/q)+1*np.ones([q,q])/q))]
            for n in n_list:
                N = int(n*np.sqrt(p*q))
                for e in e_list:
                    if dist_name == 't(1)':
                        Escale = e/np.sqrt(p*q)
                    else:
                        Escale = e*np.sqrt(np.min([p,q]))
                    df = main(size,N,dep,Fdist,Edist,sqrtSigma,sqrtOmega,Escale,rmax,sst,T=10,rep=10)
                    df.to_csv(r'./output/'+dist_name+'-%d-%d-%d-%.1f.csv'%(p,q,n,e)) #Edist-p-q-n-e.csv
                    print(dist_name+'-%d-%d-%d-%.1f.csv'%(p,q,n,e)+' Complete!')