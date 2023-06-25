# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 21:06:14 2021

@author: Alex Bradshaw
"""
import scipy  as sp
import matplotlib.pyplot as plt
import scipy.stats
A = sp.stats.norm.cdf(2) - sp.stats.norm.cdf(-2)
import math
h = 1e-6
b = 2.0
a = -2.0

hList = []


def midpoint(f,x1,x2,h):
    integral = 0
    while(x1<x2):
        height = ( f((x1+x1+h)/2) )
        integral += height * h
        x1 += h
    return integral

def trapzoid(f,x1,x2, h):
    integral = 0
    fx1 = f(x1)
    while(x1<x2): 
        fx1h = f(x1+h)
        height = (fx1 + fx1h) / (2)
        integral += height * h
        fx1 = fx1h
        x1 +=h
    return integral
        
def simpsons(f,x0,x2, h):
    mp = math.floor((x2-x0)/h)
    integral = 0
    for i in range(mp):
        x0k = x0 + h*i
        n = x0 + h*(i+0.5)
        x0kp1 = x0 + h*(i+1)
        fx0 = f(x0k)
        fn = f(n)
        fx0kp1 = f(x0kp1)    
        integral += (h/6)*(fx0 + 4*fn + fx0kp1)
    return integral

def func(x):
    return math.exp((-x**2)/2) / (math.pi*2)**(1/2)

def error(estimate):
    return abs(A - estimate)

def main():
    MList = []
    TList = []
    SList = []
    hList = []
    for i in range(2,6):
        hList.append(10**(-i))
    for h in range(len(hList)):
        MList.append(  error(  midpoint(func, a, b, hList[h])  ))
        TList.append(  error(  trapzoid(func, a, b, hList[h])  ))
        SList.append(  error(  simpsons(func, a, b, hList[h])  ))
    plt.loglog()
    plt.plot(MList,hList,color="Red",label = "Midpoint Relative Error")
    plt.plot(TList,hList,color="Blue",label = "Trapezoidal Relative Error")
    plt.plot(SList,hList,color="Green",label = "Simpson's Relative Error")
    plt.legend()
main()

