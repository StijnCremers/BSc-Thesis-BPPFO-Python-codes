import numpy as np
import time as t
import operator


def BPPFO_BFD(data, sort_indic, bin_capacity = np.inf):
    
    """
  A function that implements the First Fit Decreasing (FFD) algorithm for the bin packing problem with fragile objects.

  Parameters:
      - data (numpy array): An array, where each row contains an item's weight and fragility.
      - sort_indic (binary): An indicator, that shows if the items need to be sorted on weights or fragilities. sort_indic = 0 if to be sorted descending on weights, 
          sort_indic = 1 if to be sorted ascending on fragilities, sort_indic = 2 if to be sorted descending on fragilities.
      - bin_capacity (int): The maximum capacity of each bin. Default value is infinity for this application of the BPPFO.

  Returns:
      - num_bins (int): A list of lists, where each inner list represents a bin and contains the bin number, the bin's remaining space and the items packed in that bin.
      - runtime (float): A numerical value indicating the total time the heuristic algorithm takes to perform its process.
  """
    # Start the timer
    start = t.time()
      
    # Initialize the list of bins, the capacity vector and the number of bins
    bins = []
    bin_cap = []
    num_bins = 0
    
    # Sorting the data according to specified sort_indic
    if sort_indic == 0:
        sorted_data = data[np.lexsort([data[:, 1], -data[:, 0]])]
    elif sort_indic == 1:
        sorted_data = data[np.lexsort([-data[:, 0], data[:, 1]])]
    elif sort_indic == 2:
        sorted_data = data[np.lexsort([-data[:, 0], -data[:, 1]])]
    
    # Loop through each item and pack it into the bin with the least space left, abiding all capacity & fragility constraints
    for i in range(len(sorted_data)):
        # Set this item to not be packed
        packed = False
        
        # Sort the bins on remaining space in non-decreasing order
        bins_sort = sorted(bins, reverse=False, key=operator.itemgetter(0))
        
        # Loop through all bins starting with the one with smallest remaining space, til item is packed
        for bin in bins_sort:
            # Check if this bin has enough remaining space to fit this item
            if bin[1] >= sorted_data[i, 0]:
                # Check if this item fits in this bin, taking all fragilities of items already packed in this bin and its own into account
                if sum([x[1] for x in bin[2]]) + sorted_data[i, 0] <= min(bin_cap[bin[0] - 1], sorted_data[i, 1]):
                    # Add this item to this bin
                    item = (i + 1, sorted_data[i, 0], sorted_data[i, 1])
                    bin[2].append(item)
                    
                    # Set this bin's capacity to the minimum of the item fragilities in this bin and the given bin capacity
                    bin_cap[bin[0] - 1] = min(min([x[2] for x in bin[2]]), bin_capacity)
                    
                    # Update this bin's remaining space
                    bin[1] = bin_cap[bin[0] - 1] - sum([x[1] for x in bin[2]])
                    
                    # Set item to be packed
                    packed = True
                    break
        # If the item could not be packed in existing bins, create a new one and add the item
        if not packed:
            # Update the number of bins
            num_bins += 1
            
            # Set the new bin's capacity to the minimum of this item's fragility and the specified bin capacity
            bin_cap.append(min(bin_capacity, sorted_data[i, 1]))
            
            # Add this item to the new bin, while creating it
            item = (i + 1, sorted_data[i, 0], sorted_data[i, 1])
            bin = [num_bins, bin_cap[num_bins-1] - sorted_data[i, 0], [item]]
            bins.append(bin)
            
    
    # Calculate the runtime
    runtime = t.time() - start        

    return num_bins, runtime