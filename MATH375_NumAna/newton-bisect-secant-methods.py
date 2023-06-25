import matplotlib.pyplot as plt
import math
newValues = []
newYs = []

bisVals = []
bisX = []

secVals = []
secYs = []

eps = 1e-7
NZero = 1000000
v = 435000
nOne = 1564000


def nPrime(lamb):
        return 1000000 * math.exp(lamb) + (lamb*v*math.exp(lamb) - v*math.exp(lamb) + v)/(lamb**2)
    
def zero(lamb):
    return 1000000 * math.exp(lamb) - ((v/lamb)*(math.exp(lamb) - 1)) - 1564000


def newton(lamb):
    print("~ Start of Newton Method ~")
    flag = False
    index = 0
    while(flag==False):
        index += 1
    #for i in range(10):
        yLamb = zero(lamb)
        newValue = lamb - yLamb/(nPrime(lamb))
        newValues.append(lamb)
        newYs.append(yLamb)
        lamb = newValue
        if(abs(yLamb) < eps):
            print(lamb,yLamb, index)
            flag=True
    print("~ End of Newton Method ~") # gives lambda = 0.8
        


def bis(L, R):
    print("~ Start of Bisection Method ~")
    flag = False
    index = 0 
    while(flag == False):  
        index +=1
        half = (L+R)/2
        mid = zero(half)
        bisVals.append(mid)
        bisX.append(half)
        if(mid > 0 ):
            R = half
        elif(mid < 0 ):
            L = half
        if(abs(mid) < eps):
            print(half,mid, index)
            flag = True
    print("~ End of Bisection Method~")
    

def sec(x0, x1):
    print("~ Start of Secant Method ~")
    flag = False
    index = 0
    while(flag == False):
        index+=1
        y1 = zero(x1)
        x2 = x1 - (y1)*(x1 - x0)/(y1 - zero(x0))
        secVals.append(x1)
        secYs.append(x2)
        x0 = x1
        x1 = x2
        if(abs(y1) < eps):
            print(x1,y1, index)
            flag = True
    print("~ End of Secant Method ~")
newton(10)
bis(0.8026,0.8027)
sec(-1,5)

#print(bisVals)
plt.plot(newValues)
plt.plot(bisVals)
plt.plot(secVals)
 