import numpy as np

def Fnorm(M): #operator norm
    return np.sqrt(np.sum(M**2))

def opnorm(M): #operator norm
    return np.max(np.linalg.svd(M)[1])

def Dmetric(PA1,PA2,p0): #Dmetric
    return np.sqrt(1-np.trace(np.dot(PA1,PA2))/p0)

def projection(R):
    return np.dot(np.dot(R,np.linalg.inv(np.dot(R.T,R))),R.T)

#competing methods

def PCA2D2(data,p0,q0): #2D2PCA algorithm
    nd = len(data)
    X0 = data[0]
    EXXT = np.dot(X0,X0.T)/nd
    EXTX = np.dot(X0.T,X0)/nd
    for k in range(1,nd):
        Xk = data[k]
        EXXT += np.dot(Xk,Xk.T)/nd
        EXTX += np.dot(Xk.T,Xk)/nd
    OEA = np.linalg.svd(EXXT)[0] #2D2PCA
    A2D2 = OEA[:,:p0]
    OEB = np.linalg.svd(EXTX)[0]
    B2D2 = OEB[:,:q0]
    return A2D2,B2D2

def PE(data,p0,q0,initial,epsilon=0.01,T=10): #PE algorithm initial = [A2D2,B2D2]
    nd = len(data)
    for t in range(T): #MPCA iteration
        AM0 = initial[0]
        BM0 = initial[1]
        PAM0 = np.dot(AM0,AM0.T)
        PBM0 = np.dot(BM0,BM0.T)
        X0 = data[0]
        EXTPAX = np.dot(X0.T,np.dot(PAM0,X0))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXTPAX += np.dot(Xk.T,np.dot(PAM0,Xk))/nd
        X0 = data[0]
        EXPBXT = np.dot(X0,np.dot(PBM0,X0.T))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXPBXT += np.dot(Xk,np.dot(PBM0,Xk.T))/nd
        OEA = np.linalg.svd(EXPBXT)[0]
        OEB = np.linalg.svd(EXTPAX)[0]
        AM = OEA[:,:p0]
        BM = OEB[:,:q0]
        PAM = np.dot(AM,AM.T)
        PBM = np.dot(BM,BM.T)
        if (Fnorm(PAM-PAM0)<epsilon) and (Fnorm(PBM-PBM0)<epsilon):
            return AM,BM
        else:
            initial = [AM,BM]
    return AM,BM

def Huberweight(data,p0,q0,initial,tau='default'): 
    W = []
    AH0 = initial[0]
    BH0 = initial[1]
    PAH0 = np.dot(AH0,AH0.T)
    PBH0 = np.dot(BH0,BH0.T)
    k = [Fnorm(xs-np.dot(PAH0,np.dot(xs,PBH0))) for xs in data]
    if tau == 'default':
        tau = np.median(k)
    for xs in data:
        xc = np.dot(PAH0,np.dot(xs,PBH0))
        if Fnorm(xs-xc) <= tau:
            W.append(1/(p0*q0))
        else:
            us = np.dot(AH0.T,np.dot(xs,BH0))
            w = tau/((p0*q0)*np.sqrt(Fnorm(xs)**2-Fnorm(us)**2/(p0*q0)))
            W.append(w)
    return np.array(W)

def HuberPCA(data,p0,q0,initial,epsilon=0.01,T=10): #Huber algorithm initial = [A2D2,B2D2]
    nd = len(data)
    for t in range(T):
        AH0 = initial[0]
        BH0 = initial[1]
        W = Huberweight(data,p0,q0,initial)
        PAH0 = np.dot(AH0,AH0.T)
        PBH0 = np.dot(BH0,BH0.T)
        X0 = data[0]
        EXTPAX = W[0]*np.dot(X0.T,np.dot(PAH0,X0))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXTPAX += W[k]*np.dot(Xk.T,np.dot(PAH0,Xk))/nd
        X0 = data[0]
        EXPBXT = W[0]*np.dot(X0,np.dot(PBH0,X0.T))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXPBXT += W[k]*np.dot(Xk,np.dot(PBH0,Xk.T))/nd
        OEA = np.linalg.svd(EXPBXT)[0]
        OEB = np.linalg.svd(EXTPAX)[0]
        AH = OEA[:,:p0]
        BH = OEB[:,:q0]
        PAH = np.dot(AH,AH.T)
        PBH = np.dot(BH,BH.T)
        if (Fnorm(PAH-PAH0)<epsilon) and (Fnorm(PBH-PBH0)<epsilon):
            return AH,BH,W
        else:
            initial = [AH,BH]
        return AH,BH,W
    
#Grassmannian barycenter methods
    
def EPk_initial(data,rk): #rk is a list of individual compression dimensions
    nd = len(data)
    X0 = data[0]
    OEA,S,OEB = np.linalg.svd(X0)
    Ak = OEA[:,:rk[0]]
    Bk = OEB.T[:,:rk[0]]
    PAk = np.dot(Ak,Ak.T)/nd
    PBk = np.dot(Bk,Bk.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEA,S,OEB = np.linalg.svd(Xk)
        Ak = OEA[:,:rk[k]]
        Bk = OEB.T[:,:rk[k]]
        PAk += np.dot(Ak,Ak.T)/nd
        PBk += np.dot(Bk,Bk.T)/nd
    EPAk = PAk
    EPBk = PBk
    return EPAk,EPBk

def GB_initial(data,p0,q0,rk):
    EPAk,EPBk = EPk_initial(data,rk)
    Q = np.linalg.svd(EPAk)
    W = np.linalg.svd(EPBk)
    A_initial = Q[0][:,:p0]
    B_initial = W[0][:,:q0]
    return A_initial,B_initial

def EPAkt(data,rk,B0):
    nd = len(data)
    X0 = data[0]
    OEA = np.linalg.svd(np.dot(X0,B0))[0]
    Ak = OEA[:,:rk[0]]
    EPAk = np.dot(Ak,Ak.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEA = np.linalg.svd(np.dot(Xk,B0))[0]
        Ak = OEA[:,:rk[k]]
        EPAk += np.dot(Ak,Ak.T)/nd
    return EPAk

def EPBkt(data,rk,A0):
    nd = len(data)
    X0 = data[0]
    OEB = np.linalg.svd(np.dot(X0.T,A0))[0]
    Bk = OEB[:,:rk[0]]
    EPBk = np.dot(Bk,Bk.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEB = np.linalg.svd(np.dot(Xk.T,A0))[0]
        Bk = OEB[:,:rk[k]]
        EPBk += np.dot(Bk,Bk.T)/nd
    return EPBk

def GB_iterate(data,p0,q0,rk,initial,epsilon=0.01,T=10): #initial = [Aop,Bop]
    for t in range(1,T+1):
        A0 = initial[0]
        B0 = initial[1]
        PA0 = np.dot(A0,A0.T)
        PB0 = np.dot(B0,B0.T)
        EPAk = EPAkt(data,rk,B0)
        EPBk = EPBkt(data,rk,A0)
        A = np.linalg.svd(EPAk)[0][:,:p0]
        B = np.linalg.svd(EPBk)[0][:,:q0]
        PA = np.dot(A,A.T)
        PB = np.dot(B,B.T)
        if (Fnorm(PA-PA0)<epsilon) and (Fnorm(PB-PB0)<epsilon):
            return A,B
        else:
            initial = [A,B]
    return A,B