import gurobipy as gp
import numpy as np

# Note that you will need to have this file stored on the same path as the gurobipy folder/module.
# If it is not stored on the correct path, the code will not work.

def BPPFO_ILP(data, MaxWeight, TimeLimit = 300.0, Threads = 1):
    
    """
  A function that implements the Kantorovich model for the bin packing problem with fragile objects & provides optimal solutions that can be obtained within a reasonable amount of time.

  Parameters:
      - data (numpy array): A dataframe, where each row contains an item's weight and fragility.
      - MaxWeight (int): A numerical value that indicates the upper bound of the weight of items in the BPPFO instance. 
      - TimeLimit (float): A numerical value that indicates after how long a single optimization proces should be stopped. Default value is 300 seconds / 5 minutes.
      - Threads (int): A numerical value that indicates how many threads on the computer (the code is run on) may be used by Gurobi. Default value is 1 thread.

  Returns:
      - Several results from the optimization process. TO BE ELABORATED ON
  """
    
    # Extracting relevant information from the instance data
    N = len(data)
    w = np.array(data)[:,0]
    f = np.array(data)[:,1]
    
    
    # Suppressing additional output to fasten the code
    env = gp.Env(empty = True)
    env.setParam('OutputFlag', 0)
    env.start()

    # Creating an ILP model of the BPPFO as proposed in Clautiaux et al.
    m = gp.Model("BPPFO_ILP2")

    # Creating variables for the ILP model
    y = {}
    for i in range(N):
        y[i] = m.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}')

    x = {}
    for i in range(N):
        for j in range(i+1, N):
            x[j,i] = m.addVar(vtype=gp.GRB.BINARY, name=f'x_{j}_{i}')
        
    # Set the objective function of the ILP model
    m.setObjective(gp.quicksum(y[i] for i in range(N)), gp.GRB.MINIMIZE)

    # Adding constraints to the ILP model
    for j in range(N):
        # Adding the constraint ensuring each item is packed in exactly one bin
        m.addConstr(y[j] + gp.quicksum(x[j,i] for i in range(j-1)) == 1)
    
    for i in range(N):
        # Adding the constraint ensuring the minimum fragility in a bin is not exceeded
        m.addConstr(gp.quicksum(float(w[j]) * x[j,i] for j in range(i+1,N)) <= float(f[i] - w[i]) * y[i])
  
    for i in range(N):
        for j in range(i+1, N):
            # Adding the constraint that tightens a potential linear relaxation of the ILP model
            m.addConstr(x[j,i] <= y[i])
        
    # Setting several model parameters
    m.Params.timeLimit = TimeLimit       
    m.Params.Threads = Threads           

    # Optimizing the ILP model
    m.optimize()

    # Returning results from the Gurobi optimization
    objval = m.Objval
    if m.Status == 2:
        optimal = 'Yes'
    elif m.Status == 9:
        optimal = 'Time limit'
    else:
        optimal = 'No'
    runtime = round(m.Runtime,5)
    lb = m.MIPGap
    ub = m.Objval
    num_vars = m.NumVars
    num_cons = m.NumConstrs
    num_nonzeros = m.NumNZs
    
    # Creating the LP relaxation of the ILP model above
    Vars = m.getVars()
    pen = [1.0] * num_vars
    m.feasRelax(0, True, Vars, pen, pen, None, None)
    m.optimize()
    objval_LPRelax = m.Objval

    return  N, MaxWeight, objval, optimal, runtime, lb, ub, objval_LPRelax, num_vars, num_cons, num_nonzeros