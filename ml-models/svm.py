import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import display
import time
import psutil
import os
#%%
# Begin timing
start_time = time.time()
#%%
alpha = pd.read_csv('alph.txt', header=None, names=['alpha']).astype(int)
beta = pd.read_csv('bet.txt', header=None, names=['beta']).astype(int)
gamma = pd.read_csv('gam.txt', header=None, names=['gamma']).astype(int)
pro = pd.read_csv('pro.txt', header=None, names=['pro'])
#%%
data = pd.concat([alpha, beta, gamma, pro], axis=1)

# Replace non-finite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)
#%%
def is_power_of_2(val):
    if pd.isna(val):
        return False
    val = (int(val))
    return val == 0 or (val > 0 and (val & (val - 1)) == 0)
#%%
# Add power of 2 to dataframe
data['alpha_is_power_of_2'] = data['alpha'].apply(lambda x: is_power_of_2(int(x)))
data['beta_is_power_of_2'] = data['beta'].apply(lambda x: is_power_of_2(int(x)))
#%%
# Define optimal target
threshold = 0.0001
data['optimal'] = ((data['alpha'] == data['beta']) &
                  data['alpha_is_power_of_2'] &
                  data['beta_is_power_of_2'] &
                  (data['gamma'] == 0) | (data['gamma'].apply(lambda x: is_power_of_2(int(x)))) &
                  (data['pro'] > threshold)).astype(int)
#%%
# Features and target
features = data[['alpha', 'beta', 'gamma', 'pro']]
target = data['optimal']
#%%
# Scale features for SVM
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
#%%
#Train test split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)
#%%
# Train the SVM Classifier
svm = SVC(probability=True, random_state=42, class_weight="balanced")
svm.fit(X_train, y_train)
#%%
# Predict probabilities for the original dataset
data['predicted_prob'] = svm.predict_proba(scaler.transform(features))[:, 1]
#%%
# Post-process recommendations to ensure validity
def valid_recommendations(row):
    return (row['alpha'] == row['beta']) and \
    is_power_of_2(int(row['alpha'])) and \
    (row['gamma'] == 0 or is_power_of_2(int(row['gamma'])))
#%%
# Apply the filter
filtered_recommendations = data[data.apply(valid_recommendations, axis=1)]

# Rank by pro
#filtered_recommendations = filtered_recommendations.sort_values(by=["predicted_prob", 'pro'], ascending=False)

# Remove duplicate rows while keeping the highest predicted probability
filtered_recommendations = filtered_recommendations.drop_duplicates(subset=['alpha', 'beta', 'gamma', 'pro'])
#%%
# Display valid recommendations
print("Valid Recommendations")
display(filtered_recommendations[['alpha', 'beta', 'gamma', 'pro', 'predicted_prob']])
#%%
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time}")
#%% md
# Analysis
#%%
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
#left circular shifts
def ROL(x,r):
    #print(hex(x),hex(x))
    x = ((x >> (SIMON_TYPE - r)) + (x << r)) & mask
    #print(hex(x))
    return x

# Differential calculation functions
def ROR(x, r, mask):
    return ((x << (16 - r)) + (x >> r)) & mask
#%%
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
results = []

for _, row in filtered_recommendations.iterrows():
   
    differentials, probabilities, final_weight, log2p = simulate_rounds(int(row['alpha']), int(row['beta']), int(row['gamma']), SIMON_ROUNDS)
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

#%%

#%%
total_time = 0
recorded_times = []
cpu_usage_total = 0
mem_delta_gb_total = 0

py = psutil.Process()
for i in range(10):
    start_mem = py.memory_info().rss
    py.cpu_percent(interval=None)
    # Begin timing
    start_time = time.time()

    # Scale features for SVM
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)   
    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)
    # Train the SVM Classifier
    svm = SVC(probability=True, random_state=42, class_weight="balanced")
    svm.fit(X_train, y_train)
    # Predict probabilities for the original dataset
    data['predicted_prob'] = svm.predict_proba(scaler.transform(features))[:, 1]
    # Apply the filter
    filtered_recommendations = data[data.apply(valid_recommendations, axis=1)]

    # Rank by pro
    filtered_recommendations = filtered_recommendations.sort_values(by=["predicted_prob", 'pro'], ascending=False)

    # Remove duplicate rows while keeping the highest predicted probability
    filtered_recommendations = filtered_recommendations.drop_duplicates(subset=['alpha', 'beta', 'gamma', 'pro'])


    end_time = time.time()
    
    end_mem = py.memory_info().rss
    cpu_usage = py.cpu_percent(interval=None)

    execution_time = end_time - start_time
    
    mem_delta_gb = (end_mem - start_mem) / (1024**3)
    
    total_time += execution_time 
    cpu_usage_total += cpu_usage
    mem_delta_gb_total += mem_delta_gb
    recorded_times.append(execution_time)
    j = i + 1
#     print(f"Experiment {j} Execution time: {execution_time} | Number of recommendations: {len(filtered_recommendations)}")
    print(f"---- Experiment {j} ----")
    print(f"Execution time: {execution_time:.4f}s")
    print(f"CPU Utilisation: {cpu_usage}%")
    print(f"Memory Increment: {mem_delta_gb:.4f} GB")
    print(f"Total system RAM %: {psutil.virtual_memory().percent}%")
    print("-"*20)
    
print(f"Average time: {total_time / 10}")
print(f"Mean CPU usage: {cpu_usage_total / 10}")
print(f"Mean memory leakage: {mem_delta_gb_total / 10}")
