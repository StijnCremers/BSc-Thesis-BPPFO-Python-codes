import numpy as np
import gurobipy as gp

# Note that you will need to have this file stored on the same path as the gurobipy folder/module.
# If it is not stored on the correct path, the code will not work.

def BPPFO_ArcFlow_matheuristic(data, MaxWeight, set_div, TimeLimit = 300.0, Threads = 1):
    
    """
  A function that implements the Arc Flow model for the bin packing problem with fragile objects & provides optimal solutions that can be obtained within a reasonable amount of time.

  Parameters:
      - data (numpy array): A dataframe, where each row contains an item's weight and fragility.
      - MaxWeight (int): A numerical value that indicates the upper bound of the weight of items in the BPPFO instance. 
      - set_div (int): A numerical value that indicates in how many sets of equal fragilities we should try to divide the BPPFO instance in. '10' means the instance should be divided in 10% of the number of items subsets and '20' means 20%.
      - TimeLimit (float): A numerical value that indicates after how long a single optimization proces should be stopped. Default value is 300 seconds / 5 minutes.
      - Threads (int): A numerical value that indicates how many threads on the computer (the code is run on) may be used by Gurobi. Default value is 1 thread.
    
  Returns:
      - N (int): A numerical value indicating how many items where in the specific instance of the BPPFO.
      - Num_Graphs (int): A numerical value indicating how many graphs were created for a given instance and set_div.
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
    dataSort = data[data[:, 1].argsort()][::-1]
    dataSorted = np.copy(dataSort)
    N = len(dataSorted)
    
    
    # Deciding how to reduct the number of fragilities, based on the input set_div
    if set_div == 10:
        split = np.array_split(dataSorted, N * (set_div/100))     
    elif set_div == 20:
        split = np.array_split(dataSorted, N * (set_div/100))    
    
    
    # Re-setting the fragilities and weights of several instances, to reduce the solving time of the model
    HeurData = np.zeros([N, 2])
    
    for i in range(len(split)):
        # Set all weights to be equal to the largest weight & all fragilities
        # to be equal to the smallest fragility, if the instance remains feasible
        if max(split[i][:, 0]) > min(split[i][:, 1]):
            indic = np.argwhere(split[i][:, 0] <= min(split[i][:, 1]))
            indic_not = np.argwhere(split[i][:, 0] > min(split[i][:, 1]))
            rejected = split[i][(indic_not.T).tolist()[0], :]
            start_idx = np.argwhere(HeurData == 0)[0, 0]
            HeurData[start_idx : start_idx + len(rejected), :] = rejected
            split[i] = np.delete(split[i], (indic_not.T).tolist()[0], axis = 0)
            
            
            
            
        split[i][:, 0] = max(split[i][:, 0])
        split[i][:, 1] = min(split[i][:, 1])
        start_idx = int(np.argwhere(HeurData[:, 0] == 0)[0])
        HeurData[start_idx : start_idx + len(split[i]), :] = split[i]        
            
    HeurData = HeurData.astype('int64')
    HeurDataSort = HeurData[np.lexsort([HeurData[:, 1], - HeurData[:, 0]])]
    weight = HeurDataSort[:, 0].tolist()
    frag = HeurDataSort[:, 1].tolist()
    
    _, idx, counts = np.unique(HeurDataSort, axis = 0, return_index=True, return_counts=True)
    counts = counts[np.argsort(idx)]
    unique_rows = HeurDataSort[np.sort(idx)].tolist()
    Num_Graphs = len(unique_rows)
    GraphCap = [unique_rows[i][1] - unique_rows[i][0] for i in range(Num_Graphs)]
    
    
    
    # Suppressing additional output to fasten the code
    env = gp.Env(empty = True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    # Creating an Arc Flow model for the BPPFO
    m = gp.Model('BPPFO Arc Flow')
    
    
    # Initializing the set of active nodes in each graph
    node_set = []
    
    for b in range(Num_Graphs):
        V = np.arange(GraphCap[b] + 1)
        indic = np.r_[1, np.zeros(len(V) - 1)]
        node_set.append(np.c_[V, indic])
    
    
    # Creating a set with all possible arcs in every graph for each item
    x = {}
    item_arcs = []
    
    
    for i in range(N):
        y = {}
        
        # Iterating through the graphs the add each possible arc
        for b in range(Num_Graphs):
            
            # Deriving the initial set of active nodes in this graph
            nodes = np.where(node_set[b][:, 1] == 1)[0].tolist()
    
            # Check if the fragility of this item is larger than the fragility of the set-up item (otherwise this item should not create any arcs in this graph)
            if frag[i] >= unique_rows[b][1]:
            
                # Create arcs starting from each active arc
                for node in nodes:
                    
                    # Check if an arc from this active node with length w does not exceed the graphs capacity & check if this arc does not already exist
                    # If so, add the arc as variable to the model & set any new nodes reached to be active
                    if node + weight[i] <= GraphCap[b]:
                            y[(b, node, node + weight[i])] = None
                            node_set[b][node + weight[i], 1] = 1
                                
        # Add the dictionary of all possible arcs of this item to the item_arcs list            
        item_arcs.append(y)
        
        # Create variables corresponding to each arc added to y for this item
        for key in y:
            if key not in x:
                x[key] = m.addVar(ub = counts[key[0]] + 1, vtype = gp.GRB.INTEGER, name = f'Arc {key} corresponding to item {i}')
                
                
    # Creating the feedback and setup arcs for each graph
    for b in range(Num_Graphs):
        
        # Add this item's set-up arc
        x[(b, GraphCap[b], unique_rows[b][1])] = m.addVar(ub = counts[b], vtype = gp.GRB.INTEGER, name = f'Setup arc of graph {b}')
        
        # Add this item's feedback arc
        x[(b, unique_rows[b][1], 0)] = m.addVar(ub = counts[b], vtype = gp.GRB.INTEGER, name = f'Feedback arc of graph {b}')
        
        # Add these arcs to the corresponding item's set of arcs
        for i in range(N):
            if unique_rows[b][0] == weight[i] and unique_rows[b][1] == frag[i]:
                item_arcs[i][(b, GraphCap[b], unique_rows[b][1])] = None
                item_arcs[i][(b, unique_rows[b][1], 0)] = None
                
                                  
    # Create the loss arcs of each graph, starting from the root node (if the only active node) or the first following active node (if more nodes active)
    for b in range(Num_Graphs):
        
            for i in range(GraphCap[b]):
                x[b, i, i + 1] = m.addVar(ub = counts[b], vtype = gp.GRB.INTEGER, name = f'Loss arc ({i}, {i+1}) in graph {b}')
                
                
                
    # Set the objective function of the Arc Flow model. This is to minimize the number of bins, i.e. the number of arcs flowing out of the root node
    m.setObjective(gp.quicksum(x[(b, 0, k)] for b in range(Num_Graphs) for k in range (1, GraphCap[b] + 1) if (b, 0, k) in x), gp.GRB.MINIMIZE)
            
    
    # Add flow conservations constraints to the model
    flows = [1] + weight
    flows = list(set(flows))
    
    # For each graph, the inflow and outflow at each node should be balanced, i.e. equal
    ## Ensuring that, if some arc flowing out from the root node is traversed, then the feedback arc will be traversed as well
    for b in range(Num_Graphs):
        m.addConstr(x[b, unique_rows[b][1], 0] - gp.quicksum(x[b, 0, k] for k in flows if (b, 0, k) in x) == 0, name = 'Root nodes flow balance')
    
    for b in range(Num_Graphs):     
         
         # Balancing in- and outflow of every node between root and sink node
         V = np.arange(1, GraphCap[b]).tolist()
         for node in V:
             set_i = [node - w for w in flows if node - w >= 0]
             set_k = [node + w for w in flows if node + w <= GraphCap[b]]
             m.addConstr(gp.quicksum(x[b, i, node] for i in set_i if (b, i, node) in x) - gp.quicksum(x[b, node, k] for k in set_k if (b, node, k) in x) == 0, name = f'Node {node} flow balance in graph {b}')
         
         # Ensuring that, if some arc flowing in to the cap node is traversed, then the concluding set-up arc will be traversed as well
         set_i_capnode = [GraphCap[b] - w for w in flows if GraphCap[b] - w >= 0]
         m.addConstr(gp.quicksum(x[b, i, GraphCap[b]] for i in set_i_capnode if (b, i, GraphCap[b]) in x) - x[b, GraphCap[b], unique_rows[b][1]]  == 0, name = f'Cap node flow balance in graph {b}')
         
         # Ensuring that, if some arc flowing in to the sink node is traversed, then the feedback arc will be traversed as well
         m.addConstr(x[b, GraphCap[b], unique_rows[b][1]] - x[b, unique_rows[b][1], 0] == 0, name = f'Sink node flow balance in graph {b}')
         
         
    # Add the demand constraints to the model
    # This means that we need to have that all items are packed   
    for i in range(N):
        
        # Derive the set of demand arcs & the number of items with weight == weight[i] and fragility at least as small as frag[i]
        dem_arcs = {k: v for k, v in item_arcs[i].items() if k[2] != 0}
        w_count = 1
        
        for j in range(N):
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
    m.Params.Threads = 1
    
    
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
    
    return N, Num_Graphs, MaxWeight, objval, optimal, runtime, lb, ub, num_vars, num_cons, num_nonzeros