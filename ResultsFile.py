import numpy as np
import pandas as pd
import os
from BPPFO_ILP_function import BPPFO_ILP
from BPPFO_FFD_function import BPPFO_FFD
from BPPFO_BFD_function import BPPFO_BFD
from BPPFO_ArcFlow_function import BPPFO_ArcFlow
from BPPFO_ArcFlow_matheur import BPPFO_ArcFlow_matheuristic


# The code below tests 405 of the instances from the Clautiaux et al. (2014) benchmark set. 
# To access these instances, you will need to download them from internet (they are publicly available on http://www.or.unimore.it/resources.htm)
# and change the 'path_to_dir'-variable in line 25 accordingly.
# Similarly for the output files: you will need to change their paths to the preferred path on your on device.

# Note that you will need to have this file stored on the same path as the gurobipy folder/module.
# If it is not stored on the correct path, the code will not work.

# Initialization of parameters
TimeLimit_list = [20.0, 120.0, 300.0]

iter_num = 0
ILPresults_list = []
FFDresults_list = []
BFDresults_list = []
AFresults_list = []
AFMH10results_list = []
AFMH20results_list = []

# Importing all 675 instances into Python
path_to_dir = 'C:/Users/Stijn Cremers/OneDrive/TiU documents/Year 3/BSc Thesis/Python implementation/python311/lib/InstancesBPPFO'
filenames = os.listdir(path_to_dir)

# For each BPPFO instance: check if it should be used to test the code on, then test the code
for filename in filenames:
    iter_num += 1
    
    # Use the 1st, 3rd and 5th instance of each group of 5 instances
    if iter_num % 5 == 0 or iter_num % 5 == 3:
        # Extracting the data from this instance
        fn = f'C:/Users/Stijn Cremers/OneDrive/TiU documents/Year 3/BSc Thesis/Python implementation/python311/lib/InstancesBPPFO/{filename}'
        data = pd.read_csv(fn, sep=" ", header=None, skiprows = 2)
        data.columns = ['Weight', 'Fragility']
        NumItems_MaxWeight = pd.read_csv(fn, sep=" ", header=None, nrows = 2)
        data_arr = np.array(data)
        
        # Setting the limit on runtime of the ILP, dependent on the number of items in the instance
        TimeLimit_adj = (NumItems_MaxWeight[0][0] == 50) * TimeLimit_list[0] + (NumItems_MaxWeight[0][0] == 100) * TimeLimit_list[1] + (NumItems_MaxWeight[0][0] == 200) * TimeLimit_list[2]
        
        # Running the ILP model for this instance        
        ILPresults = BPPFO_ILP(data_arr, NumItems_MaxWeight[0][1], TimeLimit = TimeLimit_adj)
        ILPresults_list.append(ILPresults)
        
        # Running the FFD heuristic for this instance, for 3 different sorting orders
        objValue0FFD, runtime0FFD = BPPFO_FFD(data_arr, 0)
        objValue1FFD, runtime1FFD = BPPFO_FFD(data_arr, 1)
        objValue2FFD, runtime2FFD = BPPFO_FFD(data_arr, 2)
        FFDresults = (objValue0FFD, runtime0FFD, objValue1FFD, runtime1FFD, objValue2FFD, runtime2FFD)
        FFDresults_list.append(FFDresults)
        
        # Running the BFD heuristic for this instance, for 3 different sorting orders
        objValue0BFD, runtime0BFD = BPPFO_BFD(data_arr, 0)
        objValue1BFD, runtime1BFD = BPPFO_BFD(data_arr, 1)
        objValue2BFD, runtime2BFD = BPPFO_BFD(data_arr, 2)
        BFDresults = (objValue0BFD, runtime0BFD, objValue1BFD, runtime1BFD, objValue2BFD, runtime2BFD)
        BFDresults_list.append(BFDresults)
        
        # Running the Arc Flow model for this instance
        AFresults = BPPFO_ArcFlow(data_arr, NumItems_MaxWeight[0][1], TimeLimit = TimeLimit_adj)
        AFresults_list.append(AFresults)
        
        # Running the Arc Flow matheuristic for this instance
        AFMH10results = BPPFO_ArcFlow_matheuristic(data_arr, NumItems_MaxWeight[0][1], 10, TimeLimit = TimeLimit_adj)
        AFMH10results_list.append(AFMH10results)
        
        AFMH20results = BPPFO_ArcFlow_matheuristic(data_arr, NumItems_MaxWeight[0][1], 20, TimeLimit = TimeLimit_adj)
        AFMH20results_list.append(AFMH20results)



# Writing the ILP results to Excel
ILPresults_df = pd.DataFrame(ILPresults_list, columns = ["Number of items", "Upper bound on item weights", "Number of bins", "Solution optimal?", "Runtime", "Lower bound", "Upper bound", "Number of variables", "Number of constraints", "Number of nonzero elements"])
ILPresults_df.to_excel(r"C:\Users\Stijn Cremers\OneDrive\TiU documents\Year 3\BSc Thesis\Python implementation\python311\lib\BPPFO_ILP_resultsGenerated.xlsx", sheet_name="ILP Results" , index=False)

# Writing the FFD results to Excel
FFDresults_df = pd.DataFrame(FFDresults_list, columns = ["Objective value for indic = 0", "Runtime for indic = 0", "Objective value for indic = 1", "Runtime for indic = 1", "Objective value for indic = 2", "Runtime for indic = 2"])
FFDresults_df.to_excel(r"C:\Users\Stijn Cremers\OneDrive\TiU documents\Year 3\BSc Thesis\Python implementation\python311\lib\BPPFO_FFD_resultsGenerated.xlsx", sheet_name="FFD Results" , index=False)

# Writing the BFD results to Excel
BFDresults_df = pd.DataFrame(BFDresults_list, columns = ["Objective value for indic = 0", "Runtime for indic = 0", "Objective value for indic = 1", "Runtime for indic = 1", "Objective value for indic = 2", "Runtime for indic = 2"])
BFDresults_df.to_excel(r"C:\Users\Stijn Cremers\OneDrive\TiU documents\Year 3\BSc Thesis\Python implementation\python311\lib\BPPFO_BFD_resultsGenerated.xlsx", sheet_name="BFD Results" , index=False)

# Writing the Arc Flow results to Excel
AFresults_df = pd.DataFrame(AFresults_list, columns = ["Number of items", "Upper bound on item weights", "Number of bins", "Solution optimal?", "Runtime", "Lower bound", "Upper bound", "Number of variables", "Number of constraints", "Number of nonzero elements"])
AFresults_df.to_excel(r"C:\Users\Stijn Cremers\OneDrive\TiU documents\Year 3\BSc Thesis\Python implementation\python311\lib\BPPFO_AF_resultsGenerated.xlsx", sheet_name="Arc Flow Results" , index=False)

# Writing the Arc Flow Matheuristic results to Excel
AFMH10results_df = pd.DataFrame(AFMH10results_list, columns = ["Number of items", "Number of subsets", "Upper bound on item weights", "Number of bins", "Solution optimal?", "Runtime", "Lower bound", "Upper bound", "Number of variables", "Number of constraints", "Number of nonzero elements"])
AFMH10results_df.to_excel(r"C:\Users\Stijn Cremers\OneDrive\TiU documents\Year 3\BSc Thesis\Python implementation\python311\lib\BPPFO_AFMH10_resultsGenerated.xlsx", sheet_name="Matheuristic10 Results" , index=False)

AFMH20results_df = pd.DataFrame(AFMH20results_list, columns = ["Number of items", "Number of subsets", "Upper bound on item weights", "Number of bins", "Solution optimal?", "Runtime", "Lower bound", "Upper bound", "Number of variables", "Number of constraints", "Number of nonzero elements"])
AFMH20results_df.to_excel(r"C:\Users\Stijn Cremers\OneDrive\TiU documents\Year 3\BSc Thesis\Python implementation\python311\lib\BPPFO_AFMH20_resultsGenerated.xlsx", sheet_name="Matheuristic20 Results" , index=False)
