import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ortho_group
import numpy as np
from scipy import stats
import scipy.linalg
import sympy
from sstudentt import SST

def Fnorm(M): #Frobenius norm
    return np.sqrt(np.sum(M**2))

def opnorm(M): #operator norm
    return np.max(np.linalg.svd(M)[1])

def Dmetric(PA1,PA2,p0): #Dmetric
    return np.sqrt(1-np.trace(np.dot(PA1,PA2))/p0)

def projection(R):
    return np.dot(np.dot(R,np.linalg.inv(np.dot(R.T,R))),R.T)


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
    return data,PR,PC


#change q

T = 10
p = 30
q = 30
n = 30
p0 = 5
q0 = 5
ri = 3
Edist = stats.norm(0, 1)
Escale = 1
h = 0.01

F1 = []
n = 30
Q = [30,60,120,240,480]
for q in Q:
    Sscale = np.sqrt(p*q)
    Ft = []
    for t in range(T):
        try:
            data,PR,PC = simulation([p,q,p0,q0],n,ri,Sscale,Edist,Escale,h)
            nd = len(data)
            Rop,Cop = GB_initial(data,p0,q0,[ri]*nd)
            RF,CF =GB_iterate(data,p0,q0,[ri]*nd,[Rop,Cop],T=5)
            PRF = np.dot(RF,RF.T)
            Ft.append(Dmetric(PR,PRF,p0))
        except:
            print('SVD Error')
    print(str(q)+' completed.')
    F1.append(np.mean(Ft))

F2 = []
n = 60
Q = [30,60,120,240,480]
for q in Q:
    Sscale = np.sqrt(p*q)
    Ft = []
    for t in range(T):
        try:
            data,PR,PC = simulation([p,q,p0,q0],n,ri,Sscale,Edist,Escale,h)
            nd = len(data)
            Rop,Cop = GB_initial(data,p0,q0,[ri]*nd)
            RF,CF =GB_iterate(data,p0,q0,[ri]*nd,[Rop,Cop],T=5)
            PRF = np.dot(RF,RF.T)
            Ft.append(Dmetric(PR,PRF,p0))
        except:
            print('SVD Error')
    print(str(q)+' completed.')
    F2.append(np.mean(Ft))

F3 = []
n = 120
Q = [30,60,120,240,480]
for q in Q:
    Sscale = np.sqrt(p*q)
    Ft = []
    for t in range(T):
        try:
            data,PR,PC = simulation([p,q,p0,q0],n,ri,Sscale,Edist,Escale,h)
            nd = len(data)
            Rop,Cop = GB_initial(data,p0,q0,[ri]*nd)
            RF,CF =GB_iterate(data,p0,q0,[ri]*nd,[Rop,Cop],T=5)
            PRF = np.dot(RF,RF.T)
            Ft.append(Dmetric(PR,PRF,p0))
        except:
            print('SVD Error')
    print(str(q)+' completed.')
    F3.append(np.mean(Ft))

sns.set_style("whitegrid")  
sns.set(font_scale=1.5)
plt.figure(dpi=300,figsize=(3.5, 5))

sns.regplot(x=np.log(Q),y=np.log(F1),label='n=30')
sns.regplot(x=np.log(Q),y=np.log(F2),label='n=60')
sns.regplot(x=np.log(Q),y=np.log(F3),label='n=120')

plt.ylabel('log(error)')

plt.xlabel('log(q)')

plt.xticks([3,4,5,6,7])
plt.yticks([-6,-5,-4,-3])

plt.legend()

plt.savefig(r'./output/q.png', bbox_inches='tight')