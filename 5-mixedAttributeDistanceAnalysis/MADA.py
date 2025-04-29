import numpy as np
import pandas as pd

# Step 1: Define Pokémon Data
pokemon_data = pd.DataFrame({
    "Name": ["Sigilyph", "Togekiss", "Hitmontop", "Naganadel", "Dragapult", "Growlithe"],
    "Type": ["Psychic & Flying", "Fairy & Flying", "Fighting", 
             "Poison & Dragon", "Dragon & Ghost", "Fire"],  # Nominal
    "Total Stats": [490, 545, 455, 540, 600, 350],  # Numerical
    "Evolution Stage": [1, 3, 2, 2, 3, 1],  # Ordinal
    "Legendary": [0, 0, 0, 1, 0, 0]  # Binary
})

# Display Pokémon Data
print("My Pokemon with Stats:")
print(pokemon_data, "\n")

# Step 2: Calculate Nominal Distance Matrix (Hamming Distance for Type)
unique_types = {t: i for i, t in enumerate(pokemon_data["Type"].unique())}
pokemon_data["Type Code"] = pokemon_data["Type"].map(unique_types)

type_codes = pokemon_data["Type Code"].values.reshape(-1, 1)
type_dist_matrix = np.not_equal(type_codes, type_codes.T).astype(float)

print("Nominal Distance Matrix (Hamming Distance for Type):")
df_type_dist = pd.DataFrame(type_dist_matrix, index=pokemon_data["Name"], columns=pokemon_data["Name"])
df_type_dist.index.name = None
df_type_dist.columns.name = None
print(df_type_dist, "\n")

# Step 3: Define Min-Max Scaling Function
def min_max_scale(column, ordinal=False):
    min_val, max_val = column.min(), column.max()
    if ordinal:
        return (column - min_val) / (3 - 1)  # For ordinal data (m-1 scaling)
    return (column - min_val) / (max_val - min_val)  # Regular Min-Max Scaling

# Apply Scaling to Numerical and Ordinal Data
pokemon_data["Total Stats Scaled"] = min_max_scale(pokemon_data["Total Stats"])
pokemon_data["Evolution Stage Scaled"] = min_max_scale(pokemon_data["Evolution Stage"], ordinal=True)

# Step 4: Calculate Numerical Distance Matrix (Manhattan Distance for Stats)
num_data = pokemon_data["Total Stats Scaled"].values
num_dist_matrix = np.abs(num_data[:, np.newaxis] - num_data[np.newaxis, :])

print("Numerical Distance Matrix (Manhattan Distance for Stats):")
df_num_dist = pd.DataFrame(num_dist_matrix, index=pokemon_data["Name"], columns=pokemon_data["Name"])
df_num_dist.index.name = None
df_num_dist.columns.name = None
print(df_num_dist)

# Step 5: Calculate Ordinal Distance Matrix (Manhattan Distance for Evolution Stage)
ord_data = pokemon_data["Evolution Stage Scaled"].values
ord_dist_matrix = np.abs(ord_data[:, np.newaxis] - ord_data[np.newaxis, :])

print("\nOrdinal Distance Matrix (Manhattan Distance for Evolution Stage):")
df_ord_dist = pd.DataFrame(ord_dist_matrix, index=pokemon_data["Name"], columns=pokemon_data["Name"])
df_ord_dist.index.name = None
df_ord_dist.columns.name = None
print(df_ord_dist)

# Step 6: Calculate Binary Distance Matrix (Symmetric Binary for Legendary Status)
legendary_status = pokemon_data["Legendary"].values
binary_dist_matrix = np.abs(legendary_status[:, np.newaxis] - legendary_status[np.newaxis, :]).astype(float)

print("\nBinary Distance Matrix (Symmetric Binary for Legendary Status):")
df_binary_dist = pd.DataFrame(binary_dist_matrix, index=pokemon_data["Name"], columns=pokemon_data["Name"])
df_binary_dist.index.name = None
df_binary_dist.columns.name = None
print(df_binary_dist)

# Step 7: Compute Overall Distance Matrix
distance_matrices = [type_dist_matrix, num_dist_matrix, ord_dist_matrix, binary_dist_matrix]
num_attributes = len(distance_matrices)  # Number of distance matrices used

overall_distance_matrix = sum(distance_matrices) / num_attributes

print("\nOverall Distance Matrix:")
df_overall_dist = pd.DataFrame(overall_distance_matrix, index=pokemon_data["Name"], columns=pokemon_data["Name"])
df_overall_dist.index.name = None
df_overall_dist.columns.name = None
print(df_overall_dist)