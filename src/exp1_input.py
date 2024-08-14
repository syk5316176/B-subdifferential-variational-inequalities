import numpy as np
import json
#DEFINING VI(K,f)

#number of variables
N = 2
#number of constraints
M = 3

def f(x):
    return np.array([x[0]**2+x[1]-1, np.exp(x[0])-x[1]])

def Jf(x):
    return np.array([[2*x[0], 1],
                    [np.exp(x[0]),-1]])

def gi(x,i):
    if i == 0:
        residualVal = -x[0]+x[1]-1
    elif i == 1:
        residualVal =  x[0]+x[1]-1
    else:
        residualVal =  x[0]**2+x[1]-1
    return residualVal

def Jgi(x,i):
    if i == 0:
        grad = [-1,1]
    elif i == 1:
        grad =  [1,1]
    else:
        grad = [2*x[0],1]
    return np.array(grad)

def Hgi(x,i):
    if i == 0 or i == 1:
        hess = [[0,0],
                [0,0]]
    else:
        hess = [[2,0],
                [0,0]]
    return np.array(hess)

#LOADING STARTING POINTS FOR NONSMOOTH EQUATION SOLVING

with open('exp1_starting_point_NM.json', 'r') as json_file:
        startingPointList = json.load(json_file)

