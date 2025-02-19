import numpy as np 
import matplotlib.pyplot as plt

#### Simulation from distribution ####
def simu(D):
    E=np.cumsum(D)
    u=np.random.rand()
    i=1
    while (u>E[i]):
        i=i+1
    x=i-1
    return x

#### Transition simulation ####
def simu_transition(x,a,M,K,h,c,p,D):
    d=simu(D)
    print("d=", d)
    xnew=max(0,min(x+a, M)-d)
    print("x+a=", x+a)
    print("xnew=", xnew)
    reward=-K*int((a>0))-c*max(0,min(x+a,M)-x)-h*x+p*min([d, x+a, M])
    print("reward=", reward)
    return [xnew,reward]

#### Trajectory ####
def trajectory(n,x0,pi,M,K,h,c,p,D):
    X=[]
    R=[]
    X.append(x0)
    R.append(0)
    for i in range(1,n+1):
        print('X[i-1]=',X[i-1])
        a=pi[X[i-1]]
        X.append(int(simu_transition(X[i-1],a,M,K,h,c,p,D)[0]))
        R.append(simu_transition(X[i-1],a,M,K,h,c,p,D)[1])
    return [X,R]



#### Problem variables ####
M = 15
K = 0.8
h = 0.3
c = 0.5
p = 1

gamma = 0.98

#### Geometric distribution of customers ####
q = 0.1

D=[np.power(q*(1-q),i-1) for i in range(1,M+2)]
D[0]=0
D[M]=1-sum(D)



#### Markov decision process ####
def MDP(D,M):
    P=np.zeros((M+1,M+1,M+1))
    R=np.zeros((M+1,M+1))

    for a in range(0,M):
        for x in range(0,M):
            for d in range(0,M):
                P[x][Nextstate(x,a,d)][a]=P[x][Nextstate(x,a,d)][a]+D[d]
                R[x][a]= R[x][a] + D[d]*Reward(x,a,d)
    return [P,R]

def Reward(x,a,d):
    return -K*int((a>0)) - c*max(0,min(x+a,M) - x) - h*x + p*min([d , x+a , M])


def Nextstate(x,a,d):
    return max(0,min((x+a),M)-d)

    
#### Value iteration ####

def VI(P,R,gamma):
    V=[0 for i in range(M)]
    Delta=0
    while Delta<=0:
        V_=[0 for i in range(M)]
        for x in range(M):
            V_[x]=max([sum([P[x][Nextstate(x,a,d)][a]*(R[x][a]+gamma*V[x]) for d in range(M)]) for a in range(M)])
            Delta=max(Delta,abs(V[x]-V_[x]))
        V=V_
    return V


#### Policy iteration ####

def PI(P,R,gamma):
    V=[0 for i in range(M)]
    global Delta
    while Delta<=0:
        V_=[0 for i in range(M)]
        for x in range(M):
            V_[x+1]=max([sum([P[x+1][1+Nextstate(x,a,d)][a+1]*(R[1+x][1+a]+gamma*V[x+1]) for d in range(M)]) for a in range(M)])
            Delta=max(Delta,abs(V[x+1]-V_[x+1]))
        V=V_
    return V

#### policy ####
pi=2*np.ones(M+1)

#### simulation ####
x0=M
n=1000
'''
[X,R] = trajectory(n,x0,pi,M,K,h,c,p,D)

G_t_=[np.power(gamma, i)*R[i+1] for i in range(len(R)-1)]
G_t=[sum(G_t_[:i]) for i in range(len(G_t_))]
G=G_t[-1]+R[0]

print("G=", G)
#### Monte Carlo Estimation ####
MC_V=0
MC_N=100

for k in range(MC_N):
    [X,R] = trajectory(n,x0,pi,M,K,h,c,p,D)
    G=G_t[-1]+R[0]
    MC_V+=G
    
print("Monte carlo estimation of the value function:", MC_V/MC_N)




'''

[P,R]=MDP(D,M)
print("V=", VI(P,R,gamma))
#plt.plot(np.arange(0,len(R)), R)
#plt.plot(np.arange(0,len(G_t)), G_t)

#plt.show()