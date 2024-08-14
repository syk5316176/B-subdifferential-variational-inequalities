import numpy as np
import json

#DEFINING VI(K,f)

#number of variables
N = 7
#number of constraints
M = 4

def f(v):
    F = [0] * 7
    F[0] = 2*(v[0]-10)
    F[1] = 10*(v[1]-12)
    F[2] = 4*v[2]**3
    F[3] = -6*(v[3]-11)
    F[4] = 60*v[4]**5
    F[5] = 14*v[5]-4*v[6]-10
    F[6] = 4*v[6]**3-4*v[5]-8
    return np.array(F)

def Jf(v):
    return np.array([[2, 0,  0, 0, 0, 0, 0],
                     [0, 10, 0, 0, 0, 0,0 ],
                     [0, 0,  12*v[2]**2,0,0,0,0],
                     [0,0,0,-6,0,0,0],
                     [0,0,0,0,300*v[4]**4,0,0],
                     [0,0,0,0,0,14,-4],
                     [0,0,0,0,0,-4,12*v[6]**2]
                    ])
def gi(v,i):
    if i == 1:
        residualVal = 2*v[0]**2+3*v[1]**4+v[2]+4*v[3]**2+5*v[4]-127
    elif i == 2:
        residualVal = 7*v[0]+3*v[1]+10*v[2]**2+v[3]-v[4]-282
    elif i == 3:
        residualVal = 23*v[0]+v[1]**2+6*v[5]**2-8*v[6]-196
    else:
        residualVal = 4*v[0]**2+v[1]**2+2*v[2]**2+5*v[5]-11*v[6]
    return residualVal

def Jgi(v,i):
    if i == 1:
        grad = [4*v[0], 12*v[1]**3, 1, 8*v[3], 5, 0, 0]
    elif i == 2:
        grad = [7, 3, 20*v[2], 1, -1, 0, 0]
    elif i == 3:
        grad = [23, 2*v[1], 0, 0, 0, 12*v[5], -8]
    else:
        grad = [8*v[0], 2*v[1], 4*v[2], 0, 0, 5, -11]
    return np.array(grad) 

def Hgi(v,i):
    if i == 1:
        hess = [[4, 0, 0, 0, 0, 0, 0],
                [0, 36*v[1]**2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 8, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
    elif i == 2:
        hess = [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 20, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]

    elif i == 3:
        hess = [[0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 12, 0],
                [0, 0, 0, 0, 0, 0, 0]]
    else:
        hess = [[8, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0],
                [0, 0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
    return np.array(hess)

#LOADING STARTING POINTS FOR NONSMOOTH EQUATION SOLVING

with open('exp2_starting_point_NM.json', 'r') as json_file:
        startingPointList = json.load(json_file)