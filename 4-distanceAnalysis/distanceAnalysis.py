import numpy as np
import pandas as pd

# Pokémon names 
pokemon = ["Sigilyph", "Togekiss", "Hitmontop",
           "Naganadel", "Dragapult", "Growlithe"]
attributes = ["HP", "Attack", "Defense",
              "Sp. Atk", "Sp. Def", "Speed"]

stats_matrix = np.array([
[72, 58, 80, 103, 80, 97], # Sigilyph
[85, 50, 95, 120, 115, 80], # Togekiss
[50, 95, 95, 35, 110, 70], # Hitmontop
[73, 73, 73, 127, 73, 121], # Naganadel
[88, 120, 75, 100, 75, 142], # Dragapult
[55, 70, 45, 70, 50, 60] # Growlithe
])

# DataFrame
df_stats = pd.DataFrame(stats_matrix, index=pokemon, columns=attributes)

print("Numerical Attribute Matrix:")
print(df_stats)

# Distance Functions
def manhattan_distance(A, B):
    return np.sum(np.abs(A - B))
def euclidean_distance(A, B):
    return np.sqrt(np.sum((A - B) ** 2))
def supremum_distance(A, B):
    return np.max(np.abs(A - B))

# Distance Matrices Computations
num_pokemon = len(pokemon)
manhattan_matrix = np.zeros((num_pokemon, num_pokemon))
euclidean_matrix = np.zeros((num_pokemon, num_pokemon))
supremum_matrix = np.zeros((num_pokemon, num_pokemon))
for i in range(num_pokemon):
    for j in range(num_pokemon):
        manhattan_matrix[i, j] = manhattan_distance(stats_matrix[i], stats_matrix[j])
        euclidean_matrix[i, j] = euclidean_distance(stats_matrix[i], stats_matrix[j])
        supremum_matrix[i, j] = supremum_distance(stats_matrix[i], stats_matrix[j])

# Convert to DataFrame
df_manhattan = pd.DataFrame(manhattan_matrix, index=pokemon, columns=pokemon)
df_euclidean = pd.DataFrame(euclidean_matrix, index=pokemon, columns=pokemon)
df_supremum = pd.DataFrame(supremum_matrix, index=pokemon, columns=pokemon)

# Print results
print("\nManhattan Distance Matrix:")
print(df_manhattan)
print("\nEuclidean Distance Matrix:")
print(df_euclidean.to_string())
print("\nSupremum Distance Matrix:")
print(df_supremum)
