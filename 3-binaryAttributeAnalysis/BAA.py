import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', None)

data = {"Sigilyph": [0, 0, 1, 1, 0, 0, 0], 
        "Togekiss": [0, 0, 1, 0, 0, 0, 0], 
        "Hitmontop": [1, 1, 0, 0, 0, 0, 0],
        "Naganadel": [0, 1, 1, 0, 0, 0, 1],
        "Dragapult": [0, 1, 1, 0, 0, 0, 1],
        "Growlithe": [1, 1, 0, 0, 0, 1, 0]}

col = ["Bipedal",
       "Has a tail",
       "Can Levitate",
       "Has Horn",
       "Multiple Heads",
       "Has Fur",
       "Is Legendary"]

names = list(data.keys()) 
bin_matrix = np.array(list(data.values()))

full_matrix = pd.DataFrame(bin_matrix, columns=col, index=names)

def smc_distance(a, b):
    matches = np.sum(a == b)
    return 1 - (matches / len(a))

num_objects = len(bin_matrix)
dissim_matrix = np.zeros((num_objects, num_objects))

for i in range(num_objects):
    for j in range(num_objects):
        dissim_matrix[i, j] = smc_distance(bin_matrix[i], bin_matrix[j])

df_smc = pd.DataFrame(dissim_matrix, columns=names, index=names)

print("Binary Attribute Matrix:")
print(full_matrix.to_string())

print("\nDissimilarity Matrix (SMC using Numpy & Pandas):")
print(df_smc)
