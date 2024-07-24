import numpy as np

def simulation(size,RC,N,dep,Fdist,Edist,sqrtSigma,sqrtOmega,Escale): #size = [p,q,p0,q0], RC = [R,C], dep = [phi,psi], Edist = stats.t(1), sqrtOmega = [sqrtOmega1,sqrtOmega2]
    p,q,p0,q0 = size
    phi,psi = dep
    sqrtSigma1,sqrtSigma2 = sqrtSigma
    sqrtOmega1,sqrtOmega2 = sqrtOmega
    R = RC[0]
    C = RC[1]
    Ft = np.dot(sqrtSigma1,np.dot(Fdist.rvs(size=[p0,q0]),sqrtSigma2))
    Et = Escale*np.dot(sqrtOmega1,np.dot(Edist.rvs(size=[p,q]),sqrtOmega2))
    data = []
    signal = []
    for n in range(int(1.2*N)):
        Ft = phi*Ft + np.sqrt(1-phi**2)*np.dot(sqrtSigma1,np.dot(Fdist.rvs(size=[p0,q0]),sqrtSigma2))
        Et = psi*Et + np.sqrt(1-psi**2)*Escale*np.dot(sqrtOmega1,np.dot(Edist.rvs(size=[p,q]),sqrtOmega2))
        St = np.dot(R,np.dot(Ft,C.T))
        Xt = St + Et
        data.append(Xt)
        signal.append(St)
    return data[int(0.2*N):], signal[int(0.2*N):]

def simulation_sst(size,RC,N,dep,Fdist,Edist,sqrtSigma,sqrtOmega,Escale): #size = [p,q,p0,q0], RC = [R,C], dep = [phi,psi], Edist = stats.t(1), sqrtOmega = [sqrtOmega1,sqrtOmega2]
    p,q,p0,q0 = size
    phi,psi = dep
    sqrtSigma1,sqrtSigma2 = sqrtSigma
    sqrtOmega1,sqrtOmega2 = sqrtOmega
    R = RC[0]
    C = RC[1]
    Ft = np.dot(sqrtSigma1,np.dot(Fdist.rvs(size=[p0,q0]),sqrtSigma2))
    Et = Escale*np.dot(sqrtOmega1,np.dot(Edist.r([p,q]),sqrtOmega2))
    data = []
    signal = []
    for t in range(int(1.2*N)):
        Ft = phi*Ft + np.sqrt(1-phi**2)*np.dot(sqrtSigma1,np.dot(Fdist.rvs(size=[p0,q0]),sqrtSigma2))
        Et = psi*Et + np.sqrt(1-psi**2)*Escale*np.dot(sqrtOmega1,np.dot(Edist.r([p,q]),sqrtOmega2))
        St = np.dot(R,np.dot(Ft,C.T))
        Xt = St + Et
        data.append(Xt)
        signal.append(St)
    return data[int(0.2*N):], signal[int(0.2*N):]