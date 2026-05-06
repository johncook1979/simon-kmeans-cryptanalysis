import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from IPython.display import display
import time

from sklearn.metrics import silhouette_score

rand_state = 42
import psutil
import os
# Import differentials
alpha = pd.read_csv('alph.txt', header=None, names=['alpha']).astype(int)
beta = pd.read_csv('bet.txt', header=None, names=['beta']).astype(int)
gamma = pd.read_csv('gam.txt', header=None, names=['gamma']).astype(int)
pro = pd.read_csv('pro.txt', header=None, names=['pro'])
#%%
# assign diffs to dataframe
data = pd.concat([alpha, beta, gamma, pro], axis=1)

# Replace non-finite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)
#%%
# Power of 2 function
def is_pow_2(val):
    if pd.isna(val):
        return False
    val = (int(val))
    return val == 0 or (val > 0 and (val & (val - 1)) == 0)
#%%
# Check for power of 2
data['alpha_pow_2'] = data['alpha'].apply(lambda x: is_pow_2(int(x)))
data['beta_pow_2'] = data['beta'].apply(lambda x: is_pow_2(int(x)))
data['gam_pow_2'] = data['gamma'].apply(lambda x: is_pow_2(int(x)))

# Check if inputs match
data['inputs_match'] = (data['alpha'] == data['beta']) & data['alpha_pow_2']


class RoundKmeans(KMeans):
    def fit(self, X, y=None, sample_weight=None):
        #Call the original k-means fit method
        super().fit(X, y=y, sample_weight=sample_weight)
        
        rounded_centroids = []
        
        for cluster_idx, center in enumerate(self.cluster_centers_):
            
            ############
#             data = center[:-1]
#             threshold = 0.6
#             rounded_features = np.where(data % 1 >= threshold, np.ceil(data), np.floor(data))
            #######
            rounded_features = np.round(center[:-1]) # set back to round
            
            cluster_indices = np.where(self.labels_ == cluster_idx)[0]
            pro_mean = X[cluster_indices, -1].mean() if len(cluster_indices) > 0 else center[-1]
            
            rounded_centroid = np.append(rounded_features, pro_mean)
            rounded_centroids.append(rounded_centroid)
        #self.cluster_centers_ = np.round(self.cluster_centers_).astype(int)
        self.cluster_centers_ = np.array(rounded_centroids).astype(float)
        
        return self


#determine number of clusters
cluster_range = range(2, 11)
wcss = []

for k in cluster_range:
    kmeans = KMeans(n_clusters = k, random_state=rand_state)
    print(rand_state)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
dy2 = np.gradient(np.gradient(wcss))    
elbow_index = np.argmax(np.abs(dy2))

ideal_clusters = cluster_range[elbow_index]


# Plot elbow
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, wcss, marker = 'o')
# Highlight elbox with red dot
plt.plot(cluster_range[elbow_index], wcss[elbow_index], 'ro')
# Add an arrow
plt.annotate('Elbow Point', xy=(cluster_range[elbow_index], wcss[elbow_index]),
            xytext=(cluster_range[elbow_index] +2, wcss[elbow_index] +15200),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=20)
#plt.title('Silhouette Method')
plt.xlabel('Number clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.xticks(cluster_range)
#plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
plt.grid()
plt.show()


kmeans = RoundKmeans(n_clusters=ideal_clusters, random_state=rand_state)


features_array = features.to_numpy()
data['clusters'] = kmeans.fit_predict(features_array)

if np.isnan(kmeans.cluster_centers_).any():
    print("NaN found")
    print(kmeans.cluster_centers_)

for Cluster_idx, centroid in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {Cluster_idx}: {centroid}")
#%%
def calculate_distance(row, centroids):
    point = row[['alpha', 'beta', 'gamma', 'pro']].values.astype(np.float64)
    centroid = centroids[row['clusters']].astype(np.float64)
    distance = np.sqrt(np.sum((point - centroid) ** 2))
    np.nan_to_num(distance).astype(np.float64)
#     if np.isnan(distance):
    print(distance)
#     print('Point', point)
#     print('Distance', distance)
    return distance

target_cluster = 1
target_cluster_data = data[data['clusters'] == target_cluster]

target_cluster_data['distance'] = target_cluster_data.apply(
    lambda row: calculate_distance(row, kmeans.cluster_centers_[target_cluster]), 
    axis=1
)

if np.isnan(target_cluster_data['distance'].astype(np.float64)).any():
    print("NaN found")
    print(target_cluster_data['distance'])

    
data.loc[target_cluster_data.index, 'distance'] = target_cluster_data['distance'].astype(np.float64)


recommended_differentials = data[
    (data['inputs_match']) &
    data['gam_pow_2']
]

sorteded_recommendations = recommended_differentials.sort_values(by=["distance"], ascending=[True])


np.nan_to_num(x=recommended_differentials).astype(np.float64)
recommended_differentials['distance'].dropna()
# recommended_differentials.loc[target_cluster_data.index, 'distance']
recommended_differentials['distance']

np.isnan(recommended_differentials['distance'])

#%%
def calculate_distance(row, centroids):
    point = row[['alpha', 'beta', 'gamma', 'pro']].values
#    centroid = centroids[cluster_label]
    distance = np.sqrt(np.sum((point - centroid) ** 2))
    return distance

target_cluster = 1
target_cluster_data = data[data['clusters'] == target_cluster]

target_cluster_data['distance'] = target_cluster_data.apply(
    lambda row: calculate_distance(row, kmeans.cluster_centers_[target_cluster]), 
    axis=1
)

data.loc[target_cluster_data.index, 'distance'] = target_cluster_data['distance']
#data['distance'] = data.apply(lambda row: calculate_distance(row, kmeans.cluster_centers_, row['clusters']), axis=1)

display(sorteded_recommendations)

print("Cluster means (floored centroids):")
for cluster_idx, centroid in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {cluster_idx}: {centroid}")


total_time = 0
recorded_times = []
cpu_usage_total = 0
py = psutil.Process()
for i in range(10):
    # Begin timing
    start_mem = py.memory_info().rss
    py.cpu_percent(interval=None)
    start_time = time.time()
    kmeans = RoundKmeans(n_clusters=ideal_clusters, random_state=rand_state)
    data['clusters'] = kmeans.fit_predict(features_array)
    recommended_differentials = data[
        (data['inputs_match']) &
        data['gam_pow_2']
    ]
    
    end_time = time.time()

    
    
    
#     print(f"Experiment {j} Execution time: {execution_time} | Number of recommendations: {len(recommended_differentials)}")
    
    
    
    end_mem = py.memory_info().rss
    cpu_usage = py.cpu_percent(interval=None)
    
    execution_time = end_time - start_time
    
    mem_delta_gb = (end_mem - start_mem) / (1024**3)
    
    total_time += execution_time 
    cpu_usage_total += cpu_usage
    recorded_times.append(execution_time)
    j = i + 1
    
    print(f"---- Experiment {j} ----")
    print(f"Execution time: {execution_time:.4f}s")
    print(f"CPU Utilisation: {cpu_usage}%")
    print(f"Memory Increment: {mem_delta_gb:.4f} GB")
    print(f"Total system RAM %: {psutil.virtual_memory().percent}%")
    print("-"*20)
    
    
    
    # CPU and Memory usage
#     pid = os.getpid()
#     py = psutil.Process(pid)
#     current_process = psutil.Process();
#     memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
#     print('memory use:', memoryUse)
#     print(current_process.cpu_times())
#     print(psutil.virtual_memory())
    
print(f"Average time: {total_time / 10}")
print(f"Mean CPU usage: {cpu_usage_total / 10}")
print(recorded_times)


@measure_energy

total_time = 0
recorded_times = []
cpu_usage_total = 0
py = psutil.Process()
for i in range(1):
    # Begin timing
    start_mem = py.memory_info().rss
    py.cpu_percent(interval=None)
    start_time = time.time()
    kmeans = RoundKmeans(n_clusters=ideal_clusters, random_state=rand_state)
    data['clusters'] = kmeans.fit_predict(features_array)
    recommended_differentials = data[
        (data['inputs_match']) &
        data['gam_pow_2']
    ]
    
    end_time = time.time()
 
    end_mem = py.memory_info().rss
    cpu_usage = py.cpu_percent(interval=None)
    
    execution_time = end_time - start_time
    
    mem_delta_gb = (end_mem - start_mem) / (1024**3)
    
    total_time += execution_time 
    cpu_usage_total += cpu_usage
    recorded_times.append(execution_time)
    j = i + 1
    
    print(f"---- Experiment {j} ----")
    print(f"Execution time: {execution_time:.4f}s")
    print(f"CPU Utilisation: {cpu_usage}%")
    print(f"Memory Increment: {mem_delta_gb:.4f} GB")
    print(f"Total system RAM %: {psutil.virtual_memory().percent}%")
    print("-"*20)
    

print(f"Average time: {total_time / 10}")
print(f"Mean CPU usage: {cpu_usage_total / 10}")
print(recorded_times)
