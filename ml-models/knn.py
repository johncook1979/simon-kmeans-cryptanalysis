# Import libraries
import numpy as np
import scipy.stats as stats
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn import linear_model
from collections import Counter
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

import time

# Begin timing
start_time = time.time()

# Set variables

# Set mask
mask = 2**16 - 1

# define variables for SIMON round function
alpha, beta, gamma = 1, 8, 2

# DEfine the number of SIMON rounds. 
# We will be ambitious because you don't become successful without taking chances!
SIMON_ROUNDS = 20

# set the target probability (weight). Not needed in this code and can be removed
target_weight = 32

# Set the simon type
SIMON_TYPE=16
#%%
# Define the MTS (Monte-carlo Tree Search) class to store the differentials
# The original State-of-the-art (SOTA) only defined 4 objects, dx, dy, dz, wt
# Evaluating the hamming weight (hw) and probability (prob) does not increase complexity and 
# provides analytical results without needing to add additional complexities to the code base
class MTS:
    def __init__(self, dx=None, dy=None, dz=None, wt=None, hw=None, prob=None):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.wt = wt
        self.hw = hw
        self.prob = prob

        
#%%
#left circular shifts
def ROL(x,r):
    #print(hex(x),hex(x))
    x = ((x >> (SIMON_TYPE - r)) + (x << r)) & mask
    #print(hex(x))
    return x

# Differential calculation functions
def ROR(x, r, mask):
    return ((x << (16 - r)) + (x >> r)) & mask

# This function performs several crucial steps by calculating the Hamming weight of the XOR between two inputs (alpha1 and beta1)
# and checks the condition involving the third input (gamma1). The function:
# 1. XORs alpha1 and beta1, applies the mask and calculates the Hamming weight (number of 1's in the binary representation)
# 2. Checks if the results of gamma1 AND the negation of (alpha1 XOR beta1), with the mask applied, is non-zero.
# 3. If the result is non-zero the function returns a fail indicator by applying a value of (-200, 200).
# 4. Otherwise, it returns te Hamming weight and the probability, which is 2 raised to the negative Hamming weight.

def weightAND(alpha1, beta1, gamma1, mask):
    s = (gamma1 & (~(alpha1 ^ beta1))) & mask
    temp = bin((alpha1 ^ beta1) & mask)
    wt = temp[1:].count("1")
    return (-200, 200) if s != 0 else (wt, 2**(-wt))
#%%
# Function to load differentials and probabilities from text files
def highwaylist():
    # Define an empty hw list
    hw = []
    # Import the text files
    with open('alph.txt') as ip1, open('bet.txt') as ip2, open('gam.txt') as ip3, open('pro.txt') as ip4:
        for i1, i2, i3, i4 in zip(ip1, ip2, ip3, ip4):
            dx = int(i1.strip().split()[0])
            dy = int(i2.strip().split()[0])
            dz = int(i3.strip().split()[0])
            wt = float(i4.strip().split()[0])

            # Use the weightAND function to calculate the Hamming weight and probability based on the dx, dy and dz
            #precomputed_prob, _ = weightAND(dx, dy, dz, mask)
            hwt, precomputed_prob = weightAND(ROL(dx,alpha), ROL(dx,beta), dz, mask)

            # Store the differentisla along with the precomputed probability to the hw list
            hw.append(MTS(dx=dx, dy=dy, dz=dz, wt=wt, hw=hwt, prob=precomputed_prob))
        
    return hw

hw = highwaylist()

# Print the length of the hw list
print(len(hw))

# Get a count of entry, or first round probabilities.
# This will allow us to identify differentials that are of use for cryptanalysis

# Begin by initialising an empty dictionary to hold all counts
pro_dict = {}

# Loop through the differentials and count the occurances of each diff.prob value
for diff in hw:
    if diff.prob not in pro_dict:
        pro_dict[diff.prob] = 1 # Initialise the count for this value
    else:
        pro_dict[diff.prob] +=1 # increment the count for this value

# Sort the dictionary for easier correlation        
pro_dict = sorted(pro_dict.items(), key=lambda x: x[1]) 
# Print the dictionary
print(pro_dict)
#%%
# Remove differentials with a prob of 200.
# These differentials are of no value and can degrade overall performance.
# In the original SOTA code, such differentials are not stored as a best weight and the 
# heuristioc immediately selects another differential.

hw = [entry for entry in hw if entry.prob != 200]

# Print the length of the remaining class. This will show us how many rows remain the pDDT
print(len(hw))

# We will get a sense of the items in the hw class and loop over the first 50 entries
i = 0
for item in hw:
    if i < 50:
        print(item.dx, item.dy, item.dz, item.wt, item.hw, item.prob)
        i += 1


# Create a new empty dictionary
hw_dict = {}

# Loop over the objects in the hw class.
for diff in hw:
    # If the item is not yet in the class, add it for th efirst time and increment by 1
    if diff.hw not in hw_dict:
        hw_dict[diff.hw] = 1 # Initialise the count for this value
    else:
        # If the item is already in the class, simply increment the value by 1
        hw_dict[diff.hw] +=1 # increment the count for this value

# Sort the dictionary for easier correlation        
hw_dict = sorted(hw_dict.items(), key=lambda x: x[1]) 

# Print the dictionary
print(hw_dict)

# Define an empty array
X = []

# loop through each differential in the hw list and add it to X array
for entry in hw:
    X.append([entry.dx, entry.dy, entry.dz, entry.wt, entry.hw, entry.prob])

# Set X as an array
X = np.array(X)

print("Shape of X:", X.shape)
#%%
target_hw_value = 0

hw_values = X[:, 4].reshape(-1, 1) # This will extract all rows from X but only the 5th column (index 4) which is th ehamming weight (hw)

print("Shape of hw_values:", hw_values.shape)
#%%
knn = NearestNeighbors(n_neighbors=5)

knn.fit(hw_values)



target_hw = np.array([[target_hw_value]])

print("Shape of target_hw:", target_hw.shape)
#%%
distances, indices = knn.kneighbors(target_hw)

print(f"Indices of nearest items with hw closest to {target_hw_value}: {indices[0]}")
print(f"Distances from {target_hw}: {distances[0]}")

recommended_items = X[indices[0]]

for i, item in enumerate(recommended_items):
    print(f"Recommended item {i + 1}: dx={item[0]}, dy={item[1]}, dz={item[2]}, ")
#%% md
## Simulate rounds function
This function is designed to simulate the rounds and return the differential trail and corresponding round probability
#%%
# Simulate rounds
def simulate_rounds(dx, dy, dz, SIMON_ROUNDS):
    
    # Define empty lists
    final_differentials = []
    final_probabilities = []
    final_weight = []
    log2p = []

    for _ in range(SIMON_ROUNDS):
        # Use SIMON cipher round function
        temp_dx = dx
        dx = dy ^ (ROR(dx, alpha, mask) & ROR(dx, beta, mask)) ^ ROR(dx, gamma, mask)
        dy = temp_dx
        
        # Calculate the Hamming weight
        wt = bin(dx ^ temp_dx).count('1')
        
        # Track the new differentials and their associated probability
        final_differentials.append((hex(dx), hex(dy), hex(dz)))
        final_probabilities.append(2**(-wt))  # Probability
        final_weight.append(wt)
        #log2p.append(math.log2(2**(-wt)) if (2**(-wt)) > 0 else float('-inf'))
        log2p.append(-wt)
        
    return final_differentials, final_probabilities, final_weight, log2p
#%%
for i, item in enumerate(recommended_items):
    differentials, probabilities, final_weight, log2p = simulate_rounds(int(item[0]), int(item[1]), int(item[2]), SIMON_ROUNDS)
    log2psum = 0
    prob_mean = 0
    csv_table = "Round, dx, dy, dz, wt, probability, log2p\n"
    for i, (diff, prob, wt, log2p) in enumerate(zip(differentials, probabilities, final_weight, log2p)):
        dx, dy, dz = diff


        csv_table += f"{i}, {dx},{dy},{dz},{wt},{prob:.8f},{log2p}\n"
        #print(f"Round {i}: {dx}, {dy}, {dz}, Wt= {wt}, Probability = {prob}, log2p = {log2p}")
        log2psum = log2psum + (log2p)
        prob_mean += np.mean(prob)
    csv_table += f" ,  , , , ,{(prob_mean/SIMON_ROUNDS):.8f}, {log2psum}\n"
    print(csv_table)
#%%
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time}")
