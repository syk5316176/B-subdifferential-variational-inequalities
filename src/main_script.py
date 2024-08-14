from exp2_input import *   #import corresponding example input
#from exp1_input import *
from pyomo.environ import *
from pyomo.mpec import *
import numpy as np
import sympy
np.set_printoptions(edgeitems=5,linewidth=180)

#=====================================================================
#DEFINE NECESSARY FUNCTIONS AND OPTIMIZATION MODELS 

NLPSolver = SolverFactory('ipopt')  
NLPSolver.options['tol'] = 1e-11   
NewtonTol = 1e-4     #Newton method convergence tolerance  
residualTol = 1e-5   #active constraint tolerance
dualTol = 1e-5       #weakly active constraint tolerance
maxIter = 200        #maximum # Newton iterations
gamma = np.array([[1, 0], [0, 1]]) #matrix used in evaluating an LD-derivative

#define the Euclidean projector (EP)
EP = ConcreteModel()
EP.I=RangeSet(0,N-1)
EP.v = Var(EP.I)
EP.y = Param(EP.I,mutable=True,initialize=0.0)
EP.K = ConstraintList()
for i in range(0,M):
    EP.K.add(gi(EP.v,i) <= 0.0)
EP.dual=Suffix(direction=Suffix.IMPORT)
EP.obj = Objective(expr = 0.5*sum((EP.v[i]-EP.y[i])**2 for i in EP.I), sense=minimize)

def computeEP(y):
    for i in EP.I:
        EP.y[i] = y[i]
    NLPSolver.solve(EP,tee=False)
    primalSol = [value(EP.v[i]) for i in EP.I]
    dualSol = [-EP.dual[EP.K[j]] for j in range(1,M+1)]
    return primalSol, dualSol

def computeEPActiveSet(primalSol, dualSol):
    iActive = []
    iInactive = []
    iWeakActive = []
    iStrictActive = []
    constraintsResidual = {i: gi(primalSol,i) for i in range(M)}
    for i in range(0,M):
        if abs(constraintsResidual[i]) > residualTol:
            iInactive.append(i)
        else:
            iActive.append(i)
    for i in iActive:
        if abs(dualSol[i]) <= dualTol:
            iWeakActive.append(i)
        else:
            iStrictActive.append(i)
    return iActive, iStrictActive, iWeakActive, iInactive

def checkEPDiff(iActive, iWeakActive,primalSol):
    JgActive = np.array([Jgi(primalSol,i) for i in iActive])
    if np.linalg.matrix_rank(JgActive) != JgActive.shape[0]:
        return False
    elif iWeakActive != []:
        return False
    else:
        return True

def computeEPJac(iStrictActive, primalSol, dualSol):
    G = np.identity(N) + sum(dualSol[i]*Hgi(primalSol,i) for i in iStrictActive)
    JgActive = np.array([Jgi(primalSol,i) for i in iStrictActive])
    JgActiveT = np.transpose(JgActive)
    O = np.zeros((len(iStrictActive), len(iStrictActive)))
    if iStrictActive != []:
        A = np.block([[G, JgActiveT], [JgActive,O]])
    else:
        A = G
    B = np.block([[np.identity(N)], [np.zeros((len(iStrictActive), N))]])
    EPJac = np.zeros((N,N))
    for i in range(N):
        Z = np.linalg.solve(A,B[:,i])
        EPJac[:,i] = Z[0:len(Z)-len(iStrictActive)]
    return EPJac

def QPDomain(x,v,i):
    return sum(Jgi(x,i)[j]*v[j]  for j in range(N))

#define the QPs for computing LD-derivatives
EPLD = ConcreteModel()
EPLD.P = RangeSet(0,N-1)
EPLD.Q = RangeSet(0,M-1)
EPLD.v = Var(EPLD.P)
EPLD.d = Param(EPLD.P,mutable=True,initialize=0.0)
EPLD.mu = Param(EPLD.Q,mutable=True, initialize = 0.0)
EPLD.xOpt = Param(EPLD.P,mutable=True,initialize=0.0)
EPLD.matrix = Param(EPLD.P, EPLD.P, mutable = True, initialize = 0.0)
for i in range(N):
    for j in range(N):
        if i != j:
            EPLD.matrix[i,j] = sum(EPLD.mu[k]*Hgi(EPLD.xOpt,k)[i,j] for k in EPLD.Q)
        else:
            EPLD.matrix[i,j] = sum(EPLD.mu[k]*Hgi(EPLD.xOpt,k)[i,j] for k in EPLD.Q) + 1.0
EPLD.obj = Objective(expr = 0.5*sum(sum(EPLD.v[i]*EPLD.matrix[i,j] for i in EPLD.P)*EPLD.v[j] for j in EPLD.P ) - sum(EPLD.v[i]*EPLD.d[i] for i in EPLD.P), sense=minimize)
EPLD.Cone = ConstraintList()
for i in range(0,M):
    EPLD.Cone.add(QPDomain(EPLD.xOpt,EPLD.v,i) == 0.0)
for i in range(0,M):
    EPLD.Cone.add(QPDomain(EPLD.xOpt,EPLD.v,i) <= 0.0)
EPLD.dual=Suffix(direction=Suffix.IMPORT)

def computeOneQP(iWeakActive, iStrictActive, iInactive, d, EPSol, EPDual):
    for i in range(N):
        EPLD.d[i] = d[i]
        EPLD.xOpt[i] = EPSol[i]
    for i in range(M):
            EPLD.mu[i] = EPDual[i]
    mat = np.array([Jgi(EPSol,i) for i in iStrictActive])
    _, rowBase = sympy.Matrix(mat).T.rref()
    eliminatedRow = [row for row in iStrictActive if row not in rowBase]
    iStrictActive = [iStrictActive[i] for i in rowBase]
    iInactive.extend(eliminatedRow)
    for i in iInactive:
            EPLD.Cone[i+1].deactivate()
            EPLD.Cone[i+1+M].deactivate()  
    for i in iStrictActive:
        EPLD.Cone[i+1].activate()
        EPLD.Cone[i+1+M].deactivate()
    for i in iWeakActive:
        EPLD.Cone[i+1].deactivate()
        EPLD.Cone[i+1+M].activate()
    NLPSolver.solve(EPLD,tee=False)
    primalSolQP = [value(EPLD.v[i]) for i in EPLD.P]
    constraintsResidual = {}
    for i in iWeakActive:
            constraintsResidual[i] = QPDomain(EPSol,primalSolQP,i)
    iWeakActiveA = [i for i in iWeakActive if abs(constraintsResidual[i]) <= residualTol]
    iWeakActiveAS = [i for i in iWeakActiveA if abs(EPLD.dual[EPLD.Cone[i+1+M]]) > dualTol]
    iWeakActiveAW = [i for i in iWeakActiveA if i not in iWeakActiveAS]
    return primalSolQP, iStrictActive, iInactive, iWeakActiveA, iWeakActiveAS, iWeakActiveAW

def checkQPDiff(EPSol, iStrictActive, iWeakActiveA, iWeakActiveAW):
    JQPdomainActive = np.array([Jgi(EPSol,i) for i in iWeakActiveA + iStrictActive])
    if np.linalg.matrix_rank(JQPdomainActive) != JQPdomainActive.shape[0]:
        return False
    elif iWeakActiveAW != []:
        return False
    else:
        return True

def computeQPJac(iStrictActive, iWeakActiveAS,EPSol):
    G = np.array([[value(EPLD.matrix[i,j]) for j in range(N)] for i in range(N)])
    C = np.array([Jgi(EPSol,i) for i in iWeakActiveAS + iStrictActive])
    numRowC = len(iWeakActiveAS) + len(iStrictActive)
    O = np.zeros((numRowC, numRowC)) 
    if iWeakActiveAS + iStrictActive != []:
        A = np.block([[G, np.transpose(C)], [C,O]])
    else:
        A = G
    B = np.block([[np.identity(N)], [np.zeros((numRowC, N))]])
    QPJac = np.zeros((N,N))
    for i in range(N):
        Z = np.linalg.solve(A,B[:,i])
        QPJac[:,i] = Z[0:len(Z)-numRowC]
    return QPJac

def formulateNextQP(iWeakActive, iStrictActive, iInactive, iWeakActiveA, iWeakActiveAW, iWeakActiveAS):
    iInactiveNextQP = iInactive + [i for i in iWeakActive if i not in iWeakActiveA]
    iWeakActiveNextQP = iWeakActiveAW
    iStrictActiveNextQP = iStrictActive + iWeakActiveAS
    return iWeakActiveNextQP,  iStrictActiveNextQP, iInactiveNextQP

def computeNormalMapLderiv(EPSol, EPDual, iStrictActive, iWeakActive, iInactive):
    LDMatrix = np.zeros((N,N))
    for i in range(N):
        d = gamma[:,i]

        primalSolQP, iStrictActive, iInactive, iWeakActiveA, iWeakActiveAS, iWeakActiveAW = computeOneQP(iWeakActive, iStrictActive, iInactive, d, EPSol, EPDual)
        ifQPDiff = checkQPDiff(EPSol, iStrictActive, iWeakActiveA, iWeakActiveAW)
        if not ifQPDiff:
            iWeakActive, iStrictActive, iInactive = formulateNextQP(iWeakActive, iStrictActive, iInactive, iWeakActiveA, iWeakActiveAW, iWeakActiveAS)
            LDMatrix[:,i] = primalSolQP
        else:
            QPJac = computeQPJac(iStrictActive, iWeakActiveAS,EPSol)
            LDMatrix = np.matmul(QPJac,gamma)
            break
    LDnormal = np.matmul(Jf(EPSol),LDMatrix)+gamma-LDMatrix
    Lderiv = np.zeros((N,N))
    for i in range(len(LDMatrix)):
        row = np.linalg.solve(gamma.T,LDnormal[i,:])
        Lderiv[i,:] = row
    return Lderiv

def computeNormalMapJac(EPJac,primalSol):
    return np.matmul(Jf(primalSol),EPJac)+np.identity(N)-EPJac

def computeNormalMapValue(y, EPSol):
    return f(EPSol) + y - EPSol

def computeNormalMapLimitJac(EPSol, EPDual, flagQP):
    iActive, iStrictActive, iWeakActive, iInactive = computeEPActiveSet(EPSol, EPDual)
    ifDiff = checkEPDiff(iActive,iWeakActive,EPSol)
    if ifDiff:
        EPJac = computeEPJac(iStrictActive, EPSol, EPDual)
        normalMapLimitJac = np.matmul(Jf(EPSol),EPJac) + np.identity(N) - EPJac
    else:
        flagQP = True
        normalMapLimitJac = computeNormalMapLderiv(EPSol, EPDual, iStrictActive, iWeakActive, iInactive)
    return normalMapLimitJac, flagQP

def NewtonIteration(startingPoint, NewtonMethod):
    currentPoint = startingPoint
    flagConverge = True
    flagQP = False
    flagSingular = False
    for i in range(maxIter):
        EPSol, EPDual = computeEP(currentPoint)
        normalMapValue = computeNormalMapValue(currentPoint, EPSol)
        if max([abs(x) for x in normalMapValue])<=NewtonTol:
            break
        else:
            normalMapLimitJac, flagQP = computeNormalMapLimitJac(EPSol, EPDual,flagQP)
            if NewtonMethod == classicNewtonMethod:
                if np.linalg.cond(normalMapLimitJac) >= 1e6:
                    flagSingular = True
                    break
            d = NewtonMethod(normalMapLimitJac, normalMapValue)
            currentPoint = currentPoint + d
            print(f"y{i+1}:", currentPoint)
    if i == maxIter - 1:
        flagConverge = False
    return currentPoint, i, flagConverge, flagQP, flagSingular

def classicNewtonMethod(normalMapLimitJac, normalMapValue):
    d  =np.linalg.solve(normalMapLimitJac, -normalMapValue)
    return d

#define LP in LP Newton method
def keyConstraint1(model,i):
    return model.F[i]+sum(model.J[i,j]*model.zeta[j] for j in range(N)) 
def keyConstraint2(model,i):
    return model.zeta[i]
LPNewtonModel = ConcreteModel()
LPNewtonModel.P = RangeSet(0,N-1)
LPNewtonModel.zeta = Var(LPNewtonModel.P)
LPNewtonModel.gamma = Var(within=NonNegativeReals)
LPNewtonModel.F = Param(LPNewtonModel.P,mutable=True,initialize=0.0)
LPNewtonModel.J = Param(LPNewtonModel.P,LPNewtonModel.P,mutable=True,initialize=0.0)
LPNewtonModel.RHS1 = Param(mutable = True, initialize = 0.0)
LPNewtonModel.RHS2 = Param(mutable = True, initialize = 0.0)
LPNewtonModel.con = ConstraintList()
for i in range(N):
    LPNewtonModel.con.add(keyConstraint1(LPNewtonModel,i)- LPNewtonModel.gamma*LPNewtonModel.RHS1 <= 0)
    LPNewtonModel.con.add(-keyConstraint1(LPNewtonModel,i)- LPNewtonModel.gamma*LPNewtonModel.RHS1 <= 0)
    LPNewtonModel.con.add(keyConstraint2(LPNewtonModel,i) - LPNewtonModel.gamma* LPNewtonModel.RHS2<= 0)
    LPNewtonModel.con.add(-keyConstraint2(LPNewtonModel,i) - LPNewtonModel.gamma*LPNewtonModel.RHS2 <= 0)
LPNewtonModel.obj = Objective(expr = LPNewtonModel.gamma, sense=minimize)

def LPNewtonMethod(normalMapLimitJac, normalMapValue):
    for j in LPNewtonModel.P:
        LPNewtonModel.F[j] = normalMapValue[j]
        for k in LPNewtonModel.P:
            LPNewtonModel.J[j,k] = normalMapLimitJac[j][k]
    LPNewtonModel.RHS1 = max(abs(normalMapValue))**2
    LPNewtonModel.RHS2 = max(abs(normalMapValue))
    LPSolver = SolverFactory('gurobi')
    LPSolver.solve(LPNewtonModel,tee=False)
    d = np.array([value(LPNewtonModel.zeta[i]) for i in LPNewtonModel.P])
    #print("direction d = ", d)
    return d

def main(NewtonMethod):
    successRun = 0
    failedRun = 0
    print("Test", len(startingPointList), "starting point(s)")
    print("===================================================")
    for startingPoint in startingPointList:
        print("Starting point:", startingPoint)
        normalMapSol, numIter, flagConverge, flagQP, flagSingular = NewtonIteration(startingPoint,NewtonMethod)
        if flagSingular:
            print("Failed: singular sensitivity matrix encountered")
            failedRun = failedRun + 1
        else:
            if not flagConverge:
                failedRun = failedRun + 1
                print("Failed: maximum # iterations reached")
            else:
                successRun = successRun + 1
                print("Succeeded: converged within", numIter, "iterations")
            if flagQP:
                print("At least one QP was solved")
            else:
                print("no QP was solved")
        print("===================================================")
    print("# succeeded runs:", successRun)
    print("# failed runs:", failedRun)

#======================================================================================

#======================================================================================
#SOLVING NONSMOOTH EQUATION SYSTEM

if __name__ == "__main__":
    #main(LPNewtonMethod) #choose between the classicNewtonMethod and LPNewtonMethod
    main(classicNewtonMethod)