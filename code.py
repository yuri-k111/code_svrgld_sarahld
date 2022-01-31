import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.io import arff
sns.set()
%matplotlib auto

#Diabetes Data
data = arff.loadarff('./Downloads/messidor_features.arff')
df = pd.DataFrame(data[0])
print(len(df))
df.head()
one=np.ones(1151)
df = df.assign(constant = one)
cols=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','constant','18']
df=df[cols]
df.head()
data=df.to_numpy(dtype=float)
#EEG data
data = arff.loadarff('./Downloads/EEG_Eye_State.arff')
df = pd.DataFrame(data[0])
print(len(df))
df=df.replace(df.iloc[0,-1], 0)
df=df.replace(df.iloc[188,-1], 1)
df=(df-df.mean())/df.std()
one=np.ones(len(df))
df = df.assign(constant = one)
cols=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4','constant','eyeDetection']
df=df[cols]
df.head()
data=df.to_numpy(dtype=float)

def nabfi(theta,i):#\nabla f_i
    w=0
   # print(len(theta))
    for j in range (0,len(theta)):
        w+=theta[j]*data[i][j]
    #x=np.array(data[i][0:(len(data[i])-1)]*data[i][-1])
    #return -(1-(np.tanh(data[i][-1]*w))**2)*x+2*theta
    x=np.array(data[i][0:-1])*(data[i][-1]*1+(data[i][-1]-1))
    return -(1-(np.tanh((data[i][-1]*1+(data[i][-1]-1))*w))**2)*x+2*theta

def accuracy(theta,data):
    c=0
    for i in range (0,len(data)):
        w=0
        for j in range (0,len(theta)):
            w+=theta[j]*data[i][j]
        if w*(data[i][-1]*1+(data[i][-1]-1))>=0:
        #print(w,data[i][-1])
        #if w*data[i][-1]>=0:
            c+=1/len(data)
    return c

#SVRG-LD
ScoreV=[]
for epoch in range (0,100):
    theta0=np.zeros((len(data[0])-1))
    eta=0.00001
    gamma=10000
    n=len(data[0:900])
    m=int(n**0.5)
    B=int(n**0.5)
    K=400
    I=np.array(range(0,n))
    theta=theta0
    scoreV=[]
    for i in range (0,int(K/m)):
        G=0
        thetas=theta
        for j in range (0,n):
            #print(j)
            G+=nabfi(thetas,j)/n
        for j in range (0,m):
            Ik=np.random.choice(I,B,replace=False)
            nf=0
            #print(Ik)
            for k in Ik:
                nf+=(nabfi(theta,k)-nabfi(thetas,k)+G)/B
            theta=theta-eta*nf+(2*eta/gamma)**0.5*np.random.normal(size=(len(theta)))
            #print(theta)
            scoreV.append(accuracy(theta,data[900:]))
    ScoreV.append(scoreV)
#plt.plot(scoreV)

#SARAH-LD
ScoreA=[]
for epoch in range (0,100):
    theta0=np.zeros((len(data[0])-1))
    eta=0.00001
    gamma=10000
    n=len(data[0:900])
    m=int(n**0.5)
    B=int(n**0.5)
    K=400
    I= np.array(range(0,n))
    theta=theta0
    thetak1=theta0
    scoreA=[]
    for i in range (0,int(K/m)):
        G=0
        for j in range (0,n):
            #print(j)
            G+=nabfi(theta,j)/n
        thetab=theta
        theta=theta-eta*G+(2*eta/gamma)**0.5*np.random.normal(size=(len(theta)))
        nf_b=G
        for j in range (1,m):
            Ik=np.random.choice(I,B,replace=False)
            nf=0
            #print(Ik)
            for k in Ik:
                nf+=(nabfi(theta,k)-nabfi(thetab,k)+nf_b)/B
            thetab=theta
            theta=theta-eta*nf+(2*eta/gamma)**0.5*np.random.normal(size=(len(theta)))
            #print(theta)
            nf_b=nf
            scoreA.append(accuracy(theta,data[900:]))
    ScoreA.append(scoreA)
#plt.plot(scoreA)

plt.plot(np.std(ScoreV,axis=0),label="SVRG-LD")
plt.plot(np.std(ScoreA,axis=0),label="SARAH-LD")
plt.xlabel("Iterations")
plt.ylabel("Standard Deviation")
plt.legend()
plt.savefig('diabetic_std.png')
plt.show()
plt.plot(np.average(ScoreV,axis=0),label="SVRG-LD")
plt.plot(np.average(ScoreA,axis=0),label="SARAH-LD")
plt.xlabel("Iterations")
plt.ylabel("Average Accuracy")
plt.legend()
plt.savefig('diabetic_av.png')
plt.show()
