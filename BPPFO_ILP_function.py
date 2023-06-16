import numpy as np
import gurobipy as gp

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
      - N (int): A numerical value indicating how many items are in the specific instance of the BPPFO.
      - MaxWeight (int): A numerical value indicating the largest weight of all items in the specific instance of the BPPFO.
      - objval (float): A numerical value indicating the optimal solution value for the specific instance of the BPPFO.
      - optimal (string): A string that shows if the code below could solve the specific instance of the BPPFO to optimality, if it stopped because it reached the pre-determined time limit, or stopped for any other reason.
      - runtime (float): A numerical value indicating the total time the optimization model takes to perform the optimization process.
      - lb (float): A numerical value obtained from Gurobi that indicates a lower bound on the solution of the optimization model.
      - ub (float): A numerical value obtained from Gurobi that indicates an upper bound on the solution of the optimization model, which is equal to the obtained "objval".
      - num_vars (int): A numerical value indicating the number of variables in the Gurobi optimization model.
      - num_cons (int): A numerical value indicating the number of constraints in the Gurobi optimization model.
      - num_nonzeros (int): A numerical value indicating the number of nonzero entries of the constraint matrices in the Gurobi optimization model.
  """
    
  
    # Extracting relevant information from the instance data
    N = len(data)
    dataSort = data[np.lexsort([-data[:, 0], data[:, 1]])]
    w = dataSort[:, 0]
    f = dataSort[:, 1]
    
    
    # Suppressing additional output to fasten the code
    env = gp.Env(empty = True)
    env.setParam('OutputFlag', 0)
    env.start()


    # Creating an ILP model of the BPPFO as proposed in Clautiaux et al.
    m = gp.Model('BPPFO ILP')


    # Creating variables for the ILP model
    y = {}
    for i in range(N):
        y[i] = m.addVar(vtype=gp.GRB.BINARY)

    x = {}
    for i in range(N):
        for j in range(i + 1, N):
            x[j, i] = m.addVar(vtype=gp.GRB.BINARY)
        
        
    # Set the objective function of the ILP model
    m.setObjective(gp.quicksum(y[i] for i in range(N)), gp.GRB.MINIMIZE)
    

    # Adding constraints to the ILP model
    ## Adding the constraint ensuring each item is packed in exactly one bin
    for j in range(N):
        m.addConstr(y[j] + gp.quicksum(x[j, i] for i in range(j)) == 1)
    
    ## Adding the constraint ensuring the minimum fragility in a bin is not exceeded
    m.addConstrs(gp.quicksum(w[j] * x[j, i] for j in range(i + 1, N)) <= (f[i] - w[i]) * y[i] for i in range(N))
    
    ## Adding the constraint that tightens a potential linear relaxation of the ILP model
    m.addConstrs(x[j, i] <= y[i] for i in range(N) for j in range(i + 1, N))
        
    
    # Setting several model parameters
    m.Params.timeLimit = TimeLimit       
    m.Params.Threads = Threads           


    # Optimizing the ILP model
    m.optimize()

    """
    # Returning results from the Gurobi optimization
    solution = []
    for i in range(N):
        if y[i].X == 1:
            solution.append((i, i))
            for j in range(i + 1, N):
                if x[j, i].X == 1:
                    solution.append((j, i))
    """
    
                    
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
    
    
    """
    # Creating the LP relaxation of the ILP model above
    Vars = m.getVars()
    pen = [1.0] * num_vars
    m.feasRelax(0, True, Vars, pen, pen, None, None)
    m.optimize()
    objval_LPRelax = m.Objval
    """


    """
    # Check the feasibility of the solution
    ## Check if all items are packed once
    for j in range(N):
        if y[j].X != 1:
            packed = sum(x[j,i].X for i in range(j))
            if packed != 1:
                print(f'Item {j} is not packed once, but {y[j].X + packed} times')
                
    ## Check is all fragility constraints are met
    for i in range(N):
        if y[i].X == 1:
            capacity = (f[i] - w[i])
            total_packed = sum(w[j] * x[j, i].X for j in range(i + 1, N))
            if total_packed > capacity:
                print(f'The bin with item {i} as witness is too full')
    """
    

    return N, MaxWeight, objval, optimal, runtime, lb, ub, num_vars, num_cons, num_nonzeros