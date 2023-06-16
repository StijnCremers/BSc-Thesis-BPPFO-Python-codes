import numpy as np
import gurobipy as gp

# Note that you will need to have this file stored on the same path as the gurobipy folder/module.
# If it is not stored on the correct path, the code will not work.

def BPPFO_ArcFlow(data, MaxWeight, TimeLimit = 300.0, Threads = 1):
    
    """
  A function that implements the Arc Flow model for the bin packing problem with fragile objects & provides optimal solutions that can be obtained within a reasonable amount of time.

  Parameters:
      - data (numpy array): A dataframe, where each row contains an item's weight and fragility.
      - MaxWeight (int): A numerical value that indicates the upper bound of the weight of items in the BPPFO instance. 
      - TimeLimit (float): A numerical value that indicates after how long a single optimization proces should be stopped. Default value is 300 seconds / 5 minutes.
      - Threads (int): A numerical value that indicates how many threads on the computer (the code is run on) may be used by Gurobi. Default value is 1 thread.

  Returns:
      - MaxNum_Bins (int): A numerical value indicating how many items where in the specific instance of the BPPFO.
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

    # Extracting and transforming relevant information from the instance data
    dataSort = data[data[:, 0].argsort()][::-1]
    weight = dataSort[:, 0].tolist()
    frag = dataSort[:, 1].tolist()
    
    MaxNum_Bins = len(frag)
    GraphCap = [frag[i] - weight[i] for i in range(MaxNum_Bins)]
    
    
    # Suppressing additional output to fasten the code
    env = gp.Env(empty = True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    # Creating an Arc Flow model for the BPPFO
    m = gp.Model('BPPFO Arc Flow')
    
    
    # Initializing the set of active nodes in each graph
    node_set = []
    
    for b in range(MaxNum_Bins):
        V = np.arange(GraphCap[b] + 1)
        indic = np.r_[1, np.zeros(len(V) - 1)]
        node_set.append(np.c_[V, indic])
    
    
    # Creating a set with all possible arcs in every graph for each item
    x = {}
    item_arcs = []
    
    for i in range(MaxNum_Bins):
        y = {}
        
        # Add this item's set-up arc
        y[(i, GraphCap[i], frag[i])] = None
        
        # Add this item's feedback arc
        y[(i, frag[i], 0)] = None
        
        # Iterating through the graphs the add each possible arc
        for b in range(MaxNum_Bins):
            
            # Deriving the initial set of active nodes in this graph
            nodes = np.where(node_set[b][:, 1] == 1)[0].tolist()
    
            # Check if this item is not the set-up item of this graph b (its arc is already added as set-up arc)
            if i != b:
                        
                # Check if the fragility of this item is larger than the fragility of the set-up item (otherwise this item should not create any arcs in this graph)
                if frag[i] >= frag[b]:
                
                    # Create arcs starting from each active arc
                    for node in nodes:
                        
                        # Check if an arc from this active node with length w does not exceed the graphs capacity & check if this arc does not already exist
                        # If so, add the arc as variable to the model & set any new nodes reached to be active
                        if node + weight[i] <= GraphCap[b] and (b, node, node + weight[i]) not in x:
                                y[(b, node, node + weight[i])] = None
                                node_set[b][node + weight[i], 1] = 1
                                
        # Add the dictionary of all possible arcs of this item to the item_arcs list            
        item_arcs.append(y)
        
        # Create variables corresponding to each arc added to y for this item
        for key in y:
            x[key] = m.addVar(vtype = gp.GRB.BINARY, name = f'Arc {key} corresponding to item {i}')
    
                                    
    # Create the loss arcs of each graph, starting from the root node (if the only active node) or the first following active node (if more nodes active)
    for b in range(MaxNum_Bins):
        
        # Retrieve the set of active nodes for this graph
        nodes = np.where(node_set[b][:, 1] == 1)[0].tolist()
        
        # If root node is the only active one, create loss arcs starting there
        if nodes == [0]:
            for i in range(GraphCap[b]):
                x[b, i, i + 1] = m.addVar(vtype = gp.GRB.BINARY, name = f'Loss arc ({i}, {i+1}) in graph {b}')
                
        # If more nodes are active, create loss arcs starting from the 2nd active node
        else:
            for i in range(nodes[1], GraphCap[b]):
                x[b, i, i + 1] = m.addVar(vtype = gp.GRB.BINARY, name = f'Loss arc ({i}, {i+1}) in graph {b}')
                
                
                
    # Set the objective function of the Arc Flow model. This is to minimize the number of bins, i.e. the number of arcs flowing out of the root node
    m.setObjective(gp.quicksum(x[(b, 0, k)] for b in range(MaxNum_Bins) for k in range (1, GraphCap[b] + 1) if (b, 0, k) in x), gp.GRB.MINIMIZE)
            
    
    # Add flow conservations constraints to the model
    flows = [1] + weight
    flows = list(set(flows))
    
    # For each graph, the inflow and outflow at each node should be balanced, i.e. equal
    ## Ensuring that, if some arc flowing out from the root node is traversed, then the feedback arc will be traversed as well
    for b in range(MaxNum_Bins):
        m.addConstr(x[b, frag[b], 0] - gp.quicksum(x[b, 0, k] for k in flows if (b, 0, k) in x) == 0, name = 'Root nodes flow balance')
    
    for b in range(MaxNum_Bins):     
         
         # Balancing in- and outflow of every node between root and sink node
         V = np.arange(1, GraphCap[b]).tolist()
         for node in V:
             set_i = [node - w for w in flows if node - w >= 0]
             set_k = [node + w for w in flows if node + w <= GraphCap[b]]
             m.addConstr(gp.quicksum(x[b, i, node] for i in set_i if (b, i, node) in x) - gp.quicksum(x[b, node, k] for k in set_k if (b, node, k) in x) == 0, name = f'Node {node} flow balance in graph {b}')
         
         # Ensuring that, if some arc flowing in to the cap node is traversed, then the concluding set-up arc will be traversed as well
         set_i_capnode = [GraphCap[b] - w for w in flows if GraphCap[b] - w >= 0]
         m.addConstr(gp.quicksum(x[b, i, GraphCap[b]] for i in set_i_capnode if (b, i, GraphCap[b]) in x) - x[b, GraphCap[b], frag[b]]  == 0, name = f'Cap node flow balance in graph {b}')
         
         # Ensuring that, if some arc flowing in to the sink node is traversed, then the feedback arc will be traversed as well
         m.addConstr(x[b, GraphCap[b], frag[b]] - x[b, frag[b], 0] == 0, name = f'Sink node flow balance in graph {b}')
         
         
    # Add the demand constraints to the model
    # This means that we need to have that all items are packed   
    for i in range(MaxNum_Bins):
        
        # Derive the set of demand arcs & the number of items with weight == weight[i] and fragility at least as small as frag[i]
        dem_arcs = {k: v for k, v in item_arcs[i].items() if k[2] != 0}
        w_count = 1
        
        for j in range(MaxNum_Bins):
            if j != i:
                if weight[j] == weight[i] and frag[j] <= frag[i]:
                    extra_arcs = {k: v for k, v in item_arcs[j].items() if k[2] != 0}
                    dem_arcs.update(extra_arcs)
                    w_count += 1
        
        # Iterate through all possible arcs for this item, so throught the keys of the corresponding dictionary in item_arcs
        # If the item's weight is 1, the sum of used arcs from item_arcs[i] should be at least 1
        if weight[i] == 1:
            m.addConstr(gp.quicksum(x[key] for key in dem_arcs) >= w_count, name = f'Item {i} is packed at least once')
            
        # Otherwise, the sum of used arcs from item_arcs[i] should be exactly 1
        else:
            m.addConstr(gp.quicksum(x[key] for key in dem_arcs) == w_count, name = f'Item {i} is packed only once')
    
            
    
    # Setting several model parameters
    m.Params.timeLimit = 300.0
    m.Params.Threads = Threads
    
    
    # Optimizing the Arc Flow model
    m.optimize()
    
    
    """
    # Check the optimization status
    if m.status == gp.GRB.INFEASIBLE:
        # Model is infeasible
        m.computeIIS()
        m.write("model.ilp")  # Write an infeasible model file for further analysis
        print("The model is infeasible. Check 'model.ilp' for infeasible constraints.")
        
    
    # Returning results from the Gurobi optimization
    if m.Status == 2:
        solution = []
        for i in range(MaxNum_Bins):
            for k, v in item_arcs[i].items():
                    if x[k].X == 1:
                        solution.append((k[0], i))
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
    ## If every item is packed
    for i in range(MaxNum_Bins):
        w = weight[i]
        setup = sum(x[i, GraphCap[i], GraphCap[i] + w].X for b in range(MaxNum_Bins) if (b, GraphCap[b], GraphCap[b] + w) in x)
        packed = 0
        for node in range(0, max(GraphCap) - w + 1):
            pckd = sum(x[b, node, node + w].X for b in range(MaxNum_Bins) if (b, node, node + w) in x)
            packed += pckd
        if setup + packed < weight.count(w):
            print(f'Items with weight {w} are not all packed')
            
    ## If the nodes are balanced
    for b in range(MaxNum_Bins):
        
        ## If root node is balanced
        if x[b, frag[b], 0].X != sum(x[b, 0, k].X for k in range(1, GraphCap[b] + 1) if (b, 0, k) in x):
            print(f'Root node of graph {b} is unbalanced')
        
        ## If every node between root en cap is balanced
        V = np.arange(1, GraphCap[b]).tolist()
        for node in range(1, GraphCap[b]):
            inflow = sum(x[b, node - i, node].X for i in range(1, node + 1) if (b, node - i, node) in x)
            outflow = sum(x[b, node, node + k].X for k in range(1, GraphCap[b] - node + 1) if (b, node, node + k) in x)
            if inflow != outflow:
                print(f'Node {node} in graph {b} is unbalanced')
        
        ## If graph cap node is balanced
        if sum(x[b, i, GraphCap[b]].X for i in range(GraphCap[b]) if (b, i, GraphCap[b]) in x) != x[b, GraphCap[b], frag[b]].X:
            print(f'Cap node of graph {b} is unbalanced')
        
        ## If sink node is balanced
        if x[b, GraphCap[b], frag[b]].X != x[b, frag[b], 0].X:
            print(f'Sink node of graph {b} is unbalanced')
    """
    
    
    return MaxNum_Bins, MaxWeight, objval, optimal, runtime, lb, ub, num_vars, num_cons, num_nonzeros