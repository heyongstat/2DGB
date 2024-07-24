import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ortho_group
import numpy as np
from scipy import stats
import scipy.linalg
import sympy
from sstudentt import SST
import pandas as pd

def Fnorm(M): #operator norm
    return np.sqrt(np.sum(M**2))

def opnorm(M): #operator norm
    return np.max(np.linalg.svd(M)[1])

def Dmetric(PA1,PA2,p0): #Dmetric
    return np.sqrt(1-np.trace(np.dot(PA1,PA2))/p0)

def projection(R):
    return np.dot(np.dot(R,np.linalg.inv(np.dot(R.T,R))),R.T)

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

def MPCA(data,p0,q0,initial,T=50): #initial = [A2D2,B2D2]
    nd = len(data)
    AM = initial[0]
    BM = initial[1]
    for t in range(T): #MPCA iteration
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
        OEA = np.linalg.svd(EXPBXT)[0]
        OEB = np.linalg.svd(EXTPAX)[0]
        BM = OEB[:,:q0]
        AM = OEA[:,:p0]
    return AM,BM


def Huberweight(data,p0,q0,initial): #tau=median, initial = [A2D2,B2D2]
    W = []
    AH = initial[0]
    BH = initial[1]
    PAH = np.dot(AH,AH.T)
    PBH = np.dot(BH,BH.T)
    k = [Fnorm(xs-np.dot(PAH,np.dot(xs,PBH))) for xs in data]
    tau = np.median(k)
    for xs in data:
        xc = np.dot(PAH,np.dot(xs,PBH))
        if Fnorm(xs-xc) <= tau:
            W.append(1/(p0*q0))
        else:
            us = np.dot(AH.T,np.dot(xs,BH))
            w = tau/((p0*q0)*np.sqrt(Fnorm(xs)**2-Fnorm(us)**2/(p0*q0)))
            W.append(w)
    return np.array(W)

def HuberPCA(data,p0,q0,initial,T=50):
    nd = len(data)
    AH = initial[0]
    BH = initial[1]
    W = Huberweight(data,p0,q0,initial)
    for t in range(T):
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
        OEA = np.linalg.svd(EXPBXT)[0]
        OEB = np.linalg.svd(EXTPAX)[0]
        AH = OEA[:,:p0]
        BH = OEB[:,:q0]
        W = Huberweight(data,p0,q0,[AH,BH])
    return AH,BH,W

    
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

def GOE(p):
    A = np.random.normal(size=[p,p])
    return (A+A.T)/(2*np.sqrt(p))

def simulation(size,n,ri,Sscale,Edist,Escale,h): #size = [p,q,p0,q0], RC = [R,C], dep = [phi,psi], Edist = stats.t(1), sqrtOmega = [sqrtOmega1,sqrtOmega2]
    p,q,p0,q0 = size
    R = ortho_group.rvs(dim=p)[:,:p0]
    C = ortho_group.rvs(dim=q)[:,:q0]

    PR = np.dot(R,R.T)
    PC = np.dot(C,C.T)
    data = []
    signal = []
    for i in range(n):
        PRi = PR + h*GOE(p)
        PCi = PC + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        Et = Escale*Edist.rvs(size=[p,q])
        Xt = St + Et
        data.append(Xt)
        signal.append(St)
    return data,signal

p = 100
q = 100
n = 100
p0 = 10
q0 = 10
ri = 5

Sscale = np.sqrt(p*q)
Edist = stats.norm(0, 1)
Escale = 1
h = 0.2


def MPE(data,PR,PC):
    mpe = 0
    for i in range(len(data)):
        mpe += (Fnorm(data[i]-PR@data[i]@PC)/Fnorm(data[i]))**2
    return mpe/len(data)

T = 10

PE = pd.DataFrame()
Huber = pd.DataFrame()
IGB = pd.DataFrame()


for outnum in [5,10,15,20,25,30,35,40,45,50]:
    MPEM = []
    MPEH = []
    MPEF = []

    for t in range(T):
        try:
            data,signal = simulation([p,q,p0,q0],n-outnum,ri,Sscale,Edist,Escale,h)
            data2,signal2 = simulation([p,q,p0,q0],outnum,ri,np.sqrt(0.1*n)*Sscale,Edist,Escale,h)
            dataout = data+data2

            R2D2,C2D2 = PCA2D2(dataout,p0,q0)
    
            RM,CM = MPCA(dataout,p0,q0,[R2D2,C2D2],T=5)
            PRM = np.dot(RM,RM.T)
            PCM = np.dot(CM,CM.T)    

            RH,CH,W = HuberPCA(dataout,p0,q0,[R2D2,C2D2],T=5)
            PRH = np.dot(RH,RH.T)
            PCH = np.dot(CH,CH.T) 
            
            nd = len(dataout)

            Rop,Cop = GB_initial(dataout,p0,q0,[ri]*nd)
            RF,CF = GB_iterate(dataout,p0,q0,[ri]*nd,[Rop,Cop],T=5)
            PRF = np.dot(RF,RF.T)
            PCF = np.dot(CF,CF.T)

            MPEM.append(MPE(data,PRM,PCM))
            MPEH.append(MPE(data,PRH,PCH))
            MPEF.append(MPE(data,PRF,PCF))
        except:
            print('SVD Error.')
    PE[str(outnum)+'%'] = MPEM
    Huber[str(outnum)+'%'] = MPEH
    IGB[str(outnum)+'%'] = MPEF


plt.figure(dpi=250,figsize=(11,3)) #设定图像尺寸
plt.ylim((0, 1.1))
plt.title('Projection estimator',fontsize=13)
PE.boxplot()
plt.savefig(r'./output/PE.png', bbox_inches='tight')

plt.figure(dpi=250,figsize=(11,3)) #设定图像尺寸
plt.ylim((0, 1.1))
plt.title('Huber estimator',fontsize=13)
Huber.boxplot()
plt.savefig(r'./output/Huber.png', bbox_inches='tight')

plt.figure(dpi=250,figsize=(11,3)) #设定图像尺寸
plt.ylim((0, 1.1))
plt.title('Grassmannian estimator',fontsize=13)
IGB.boxplot()
plt.savefig(r'./output/GB.png', bbox_inches='tight')