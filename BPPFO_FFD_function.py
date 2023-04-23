import numpy as np
import time as t

def BPPFO_FFD(data, sort_indic, bin_capacity = np.inf):
    
    """
  A function that implements the First Fit Decreasing (FFD) algorithm for the bin packing problem with fragile objects.

  Parameters:
      - data (numpy array): An array, where each row contains an item's weight and fragility.
      - sort_indic: An indicator, that shows if the items need to be sorted on weights or fragilities. sort_indic = 0 if to be sorted descending on weights, 
          sort_indic = 1 if to be sorted ascending on fragilities, sort_indic = 2 if to be sorted descending on fragilities.
      - bin_capacity (int): The maximum capacity of each bin. Default value is infinity for this application of the BPPFO.

  Returns:
      - bins: A list of lists, where each inner list represents a bin and contains the items packed in that bin.
  """
    # Start the timer
    start = t.time()
      
    # Initialize the list of bins and the bin capacity vector
    bins = [[]]
    bin_cap = [bin_capacity]
    
    # Sorting the data according to specified sort_indic
    if sort_indic == 0:
        sorted_data = data[np.lexsort([data[:, 1], -data[:, 0]])]
    elif sort_indic == 1:
        sorted_data = data[np.lexsort([-data[:, 0], data[:, 1]])]
    elif sort_indic == 2:
        sorted_data = data[np.lexsort([-data[:, 0], -data[:, 1]])]
    
    # Loop through each item and pack it into the first bin with enough space left
    for i in range(len(sorted_data)):
        # Set this item to not be packed
        packed = False
        
        # Find first bin that can fit this item
        for j in range(len(bins)):
            # Check if this item fits in this bin, taking all weights and fragilities of items already packed in this bin and its own into account
            if sum([x[1] for x in bins[j]]) + sorted_data[i, 0] <= min(bin_cap[j], sorted_data[i, 1]):
                # Add this item to this bin
                item = (i + 1, sorted_data[i, 0], sorted_data[i, 1])
                bins[j].append(item)
                
                # Set this bin's capacity to the minimum of the item fragilities in this bin and the given bin capacity
                bin_cap[j] = min(min([x[2] for x in bins[j]]), bin_capacity)
                
                # Set item to be packed
                packed = True
                break
        # If the item could not be packed in existing bins, create a new one and add the item
        if not packed:
            # Add this item to the new bin, while creating it
            item = (i + 1, sorted_data[i, 0], sorted_data[i, 1])
            bins.append([item])
            
            # Set this bin's capacity to the minimum item fragility in this bin
            bin_cap.append(min(bin_capacity, sorted_data[i, 1]))
            
    # Calculate the total remaining empty space squared
    RemainCap = sum([(bin_cap[bins.index(j)] - sum([x[1] for x in j])) ** 2 for j in bins]) 
    
    # Calculate the objective value
    objValue = len(bins)
            
    # Calculate the runtime
    runtime = t.time() - start
                
    return objValue, runtime, RemainCap