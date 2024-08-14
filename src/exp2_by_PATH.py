from pyomo.environ import *
from pyomo.mpec import *
import json

with open("exp2_starting_point_VI.json", "r") as f:
     startingPointListU = json.load(f)

# Create the Pyomo models with different starting points in a loop
modelList = []
for i in range(len(startingPointListU)):
    model = ConcreteModel()

    model.P=RangeSet(1,7)
    initialForU = {j: startingPointListU[i][j-1] for j in model.P}
    #primal variables
    model.u = Var(model.P,initialize=initialForU)
    model.D=RangeSet(1,4)
    #dual variables
    model.w=Var(model.D,initialize = 0.001)   #assign the initial value 0, 0.0001, 0.001, 0.01, 0.1, or 1 

    model.con1 = Complementarity(expr=complements( model.u[1], 2*(model.u[1]-10)+model.w[1]*4*model.u[1]+model.w[2]*7+model.w[3]*23+model.w[4]*8*model.u[1] == 0))
    model.con2 = Complementarity(expr=complements(model.u[2], 10*(model.u[2]-12)+model.w[1]*12*model.u[2]**3+model.w[2]*3+model.w[3]*2*model.u[2]+model.w[4]*2*model.u[2] == 0))
    model.con3 = Complementarity(expr=complements(model.u[3], 4*model.u[3]**3+model.w[1]+model.w[2]*20*model.u[3]+model.w[4]*4*model.u[3] == 0))
    model.con4 = Complementarity(expr=complements(model.u[4], -6*(model.u[4]-11)+model.w[1]*8*model.u[4]+model.w[2] == 0))                          
    model.con5 = Complementarity(expr=complements(model.u[5], 60*model.u[5]**5+model.w[1]*5-model.w[2] == 0))
    model.con6 = Complementarity(expr=complements(model.u[6], 14*model.u[6]-4*model.u[7]-10+model.w[3]*12*model.u[6]+model.w[4]*5 == 0))
    model.con7 = Complementarity(expr=complements(model.u[7], 4*model.u[7]**3-4*model.u[6]-8-model.w[3]*8-model.w[4]*11 == 0))

    model.cond1 = Complementarity(expr=complements(model.w[1]>=0, -(2*model.u[1]**2+3*model.u[2]**4+model.u[3]+4*model.u[4]**2+5*model.u[5]-127) >= 0))
    model.cond2 = Complementarity(expr=complements(model.w[2]>=0, -(7*model.u[1]+3*model.u[2]+10*model.u[3]**2+model.u[4]-model.u[5]-282) >= 0))
    model.cond3 = Complementarity(expr=complements(model.w[3]>=0, -(23*model.u[1]+model.u[2]**2+6*model.u[6]**2-8*model.u[7]-196) >= 0))
    model.cond4 = Complementarity(expr=complements(model.w[4]>=0, -(4*model.u[1]**2+model.u[2]**2+2*model.u[3]**2+5*model.u[6]-11*model.u[7]) >= 0))
    # Append the model to the list
    modelList.append(model)

success_run = 0
failed_run=0

for i, model in enumerate(modelList, start=1):
    print(f"{i}th starting point:")
    solver = SolverFactory('pathampl')
    solver_options = {'convergence_tolerance': 1e-4}
    try:
        solver.solve(model, tee=False,options = solver_options)
        success_run = success_run+1
        print("Converged")
        primalSol = {}
        dualSol = {}
        for i in model.P:
            primalSol[i]= value(model.u[i])
        for i in model.D:
            dualSol[i] = value(model.w[i])
        print("Primal solution:", primalSol)
        print("Dual solution:", dualSol)
    except:
        print("Running failed")
        failed_run=failed_run+1

print("# failed_run:",failed_run)
print("# success run:", success_run)
