#Project 2 Math 375
import math
import numpy as np
import matplotlib.pyplot as plt

hList = []
hFDVals = []
hCDVals = []
fddVals = []
gList = []

for i in range(4,7):
    hList.append(1*10**(-1*i))
                 

def forward_difference(f,x,h):
    return (f(x+h) - f(x)) / h

def centered_difference(f,x,h):
    return (f(x+h) - f(x-h)) / (2*h)

def compute_second_derivative(f,x,h):
    f1 = centered_difference(f, x, h)
    f2 = centered_difference(f, x+h, h)
    return (f2 - f1) / h

def func2(x):
    return math.sin(x[0]-x[1])

def func(x):
    return x*math.exp(x)

def fPrime(x):
    return (x+1.0)*math.exp(x)   
 
def fdPrime(x):
    return (x+2.0)*math.exp(x)    

def compute_gradient(f,x,h):
    newX = x.copy()
    for i in range(len(x)):
        vecUp = x.copy()
        vecLow = x.copy()
        vecUp[i] = vecUp[i]+h
        vecLow[i] = vecLow[i]-h
        #print(vecUp,vecLow)
        newX[i] = (f(vecUp) - f(vecLow)) / (2*h)  
    return newX

def mag(vec):
    s = 0
    for i in range(len(vec)):
        s += vec[i]**2
    return s**(1/2)

def error(x,gradient):
    newL = []
    for i in range(len(x)):
        newL.append(x[i]-gradient[i])
    #print(mag(gradient))
    return mag(newL)/mag(gradient)


d = fPrime(2.0)
dd = fdPrime(2.0)

x = [2.0,3.0]
y = [math.cos(1),-1*math.cos(1)]
for j in range(len(hList)):
    gList.append(error(y,compute_gradient(func2, x, hList[j])))
    #gList.append(compute_gradient(func2, x, hList[j]))
    #fddVals.append(dd - compute_second_derivative(func, 2, hList[j]))
    #print(hList[j],fddVals[j])
    #hFDVals.append( d - forward_difference(func, 2, hList[j]))
    #hCDVals.append( d - centered_difference(func, 2, hList[j]))


plt.plot(hList,gList,color="purple",label = "Relative Error")
#plt.semilogy makes the y axis logorithmic
#plt.plot(hList,fddVals,color="green",label = "Second Derivative")
plt.legend()
#plt.plot(hList,hFDVals,color="red")
#plt.plot(hList,hCDVals,color="blue")