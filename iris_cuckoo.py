import numpy as np
from sklearn import datasets
import pandas as pd
import math

iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target

nests = data.iloc[:3].values
all_data = data.values

def fitness(nest):
    mean = np.mean(all_data, axis=0)
    return np.linalg.norm(nest - mean)

n_nests = 3
n_iterations = 2
pa = 0.25

def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.randn(len(nests[0])) * sigma
    v = np.random.randn(len(nests[0]))
    step = u / (np.abs(v) ** (1 / Lambda)) 
    return step

for iteration in range(n_iterations):
    print(f"Iteration {iteration+1}")
    new_nests = nests.copy()
    for i in range(n_nests):
        step_size = levy_flight(1.5)
        new_nest = nests[i] + step_size * (nests[i] - np.mean(nests, axis=0))
        new_nest = np.clip(new_nest, np.min(all_data, axis=0), np.max(all_data, axis=0))
        if fitness(new_nest) < fitness(nests[i]):
            new_nests[i] = new_nest
        print(f" Nest {i+1}: {new_nests[i]}, Fitness: {fitness(new_nests[i]):.4f}")
    for i in range(n_nests):
        if np.random.rand() < pa:
            rand_idx = np.random.randint(0, all_data.shape[0])
            new_nests[i] = all_data[rand_idx].copy()  
            print(f"  Nest {i+1} abandoned and replaced with record {rand_idx}")
    nests = new_nests.copy()  

print("\nFinal nests after 2 iterations:")
for i, nest in enumerate(nests):
    print(f"Nest {i+1}: {nest}, Fitness: {fitness(nest):.4f}")
