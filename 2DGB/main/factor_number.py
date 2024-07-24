import numpy as np
from algorithms import *

#competing methods

def ER2D2(data,rmax):
    nd = len(data)
    X0 = data[0]
    EXXT = np.dot(X0,X0.T)/nd
    EXTX = np.dot(X0.T,X0)/nd
    for k in range(1,nd):
        Xk = data[k]
        EXXT += np.dot(Xk,Xk.T)/nd
        EXTX += np.dot(Xk.T,Xk)/nd
    DA = np.linalg.svd(EXXT)[1]
    DB = np.linalg.svd(EXTX)[1]
    p2D2 = np.argmax([DA[i]/DA[i+1] for i in range(rmax)])+1
    q2D2 = np.argmax([DB[i]/DB[i+1] for i in range(rmax)])+1
    return p2D2,q2D2

def ERPE(data,rmax,T=10):
    nd = len(data)
    pM0,qM0 = ER2D2(data,rmax)
    for t in range(T):
        AM,BM = PE(data,pM0,qM0,PCA2D2(data,pM0,qM0),T=T,epsilon=0.01)
        PAM = np.dot(AM,AM.T)
        PBM = np.dot(BM,BM.T)
        X0 = data[0]
        EXTPAX = np.dot(X0.T,np.dot(PAM,X0))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXTPAX += np.dot(Xk.T,np.dot(PAM,Xk))/nd
        X0 = data[0]
        EXPBXT = np.dot(X0,np.dot(PBM,X0.T))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXPBXT += np.dot(Xk,np.dot(PBM,Xk.T))/nd
        DA = np.linalg.svd(EXPBXT)[1]
        DB = np.linalg.svd(EXTPAX)[1]
        pM = np.argmax([DA[i]/DA[i+1] for i in range(rmax)])+1
        qM = np.argmax([DB[i]/DB[i+1] for i in range(rmax)])+1
        if (pM==pM0) and (qM==qM0):
            return pM,qM
        else:
            pM0 = pM
            qM0 = qM
    return pM,qM

def ERHuber(data,rmax,T=10):
    nd = len(data)
    pH0,qH0 = ER2D2(data,rmax)
    for t in range(T):
        AH,BH,W = HuberPCA(data,pH0,qH0,PCA2D2(data,pH0,qH0),T=T,epsilon=0.01)
        PAH = np.dot(AH,AH.T)
        PBH = np.dot(BH,BH.T)
        X0 = data[0]

        EXTPAX = W[0]*np.dot(X0.T,np.dot(PAH,X0))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXTPAX += W[k]*np.dot(Xk.T,np.dot(PAH,Xk))/nd
        X0 = data[0]
        EXPBXT = W[0]*np.dot(X0,np.dot(PBH,X0.T))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXPBXT += W[k]*np.dot(Xk,np.dot(PBH,Xk.T))/nd
        DA = np.linalg.svd(EXPBXT)[1]
        DB = np.linalg.svd(EXTPAX)[1]
        pH = np.argmax([DA[i]/DA[i+1] for i in range(rmax)])+1
        qH = np.argmax([DB[i]/DB[i+1] for i in range(rmax)])+1
        if (pH==pH0) and (qH==qH0):
            return pH,qH
        else:
            pH0 = pH
            qH0 = qH
    return pH,qH

#Grassmannian barycenter methods

def Er(data,rmax):
    Er = 0
    for Xs in data:
        DX = np.linalg.svd(Xs)[1]
        r = np.argmax([DX[i]/DX[i+1] for i in range(rmax)])+1
        Er += r
    return int(np.round(Er/len(data)))

def ER_initial(data,rmax,r=0):
    if r ==0:
        r = Er(data,rmax) #we choose the same r by Er for simplicity
    nd = len(data)
    rk = [r]*nd
    EPAk,EPBk = EPk_initial(data,rk)
    DA = np.linalg.svd(EPAk)[1]
    DB = np.linalg.svd(EPBk)[1]
    p_initial = np.argmax([DA[i]/DA[i+1] for i in range(rmax)])+1
    q_initial = np.argmax([DB[i]/DB[i+1] for i in range(rmax)])+1
    return p_initial,q_initial

def ER_iterate(data,rmax,r=0,T=10):
    pF0,qF0 = ER_initial(data,rmax,r=r)
    nd = len(data)
    if r == 0:
        r = Er(data,rmax) #we choose the same r by Er for simplicity
    rk = [r]*nd
    for t in range(T):
        AF,BF = GB_iterate(data,pF0,qF0,rk,GB_initial(data,pF0,qF0,rk),T=T,epsilon=0.01)
        EPAk = EPAkt(data,[np.min([r,qF0])]*nd,BF)
        EPBk = EPBkt(data,[np.min([r,pF0])]*nd,AF)
        DA = np.linalg.svd(EPAk)[1]
        DB = np.linalg.svd(EPBk)[1]
        pF = np.argmax([DA[i]/DA[i+1] for i in range(rmax)])+1
        qF = np.argmax([DB[i]/DB[i+1] for i in range(rmax)])+1
        if (pF==pF0) and (qF==qF0):
            return pF,qF
        else:
            pF0 = pF
            qF0 = qF
    return pF,qF