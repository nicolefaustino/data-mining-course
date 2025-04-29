import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data for six Pokémon
data = {
    'Pokemon': ['Sigilyph', 'Togekiss', 'Hitmontop', 'Naganadel', 'Dragapult', 'Growlithe'],
    'HP': [72, 85, 50, 73, 88, 55],
    'Attack': [58, 50, 95, 73, 120, 70],
    'Defense': [80, 95, 95, 73, 75, 45],
    'Speed': [103, 120, 35, 127, 100, 70],
    'Sp. Atk': [80, 115, 110, 73, 75, 50],
    'Sp. Def': [97, 80, 70, 121, 142, 60] }

# Create DataFrame and set index
pokemon_df = pd.DataFrame(data)
pokemon_df.set_index('Pokemon', inplace=True)

# Function to compute cosine similarity
def cosine_similarity_matrix(df):
    stats = df.values
    norm = np.linalg.norm(stats, axis=1)
    similarity_matrix = np.dot(stats, stats.T) / (norm[:, None] * norm[None, :])
    return pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

# Print the Data Matrix
print("My Pokemon Data Matrix:")
print(pokemon_df, "\n")

# Calculate cosine similarity and convert to angles in degrees
cosine_sim_df = cosine_similarity_matrix(pokemon_df)
angles_rad = np.arccos(np.clip(cosine_sim_df.values, -1.0, 1.0))
angles_deg = np.degrees(angles_rad)
angular_sim_df = pd.DataFrame(angles_deg, index=cosine_sim_df.index, columns=cosine_sim_df.columns)

# Remove the index name "Pokemon"
cosine_sim_df.index.name = None
cosine_sim_df.columns.name = None

angular_sim_df.index.name = None

# Print Cosine Similarity Matrix in Decimal Form
print("Cosine Similarity Matrix (Decimals):")
print(cosine_sim_df.round(2), "\n")  # Rounded to 4 decimal places for readability

print("Cosine Similarity Matrix (Degrees):")
print(angular_sim_df.round(2))

# List Pokémon names and extract unique pairs (upper triangle)
pokemon_names = angular_sim_df.index.tolist()
pairs = [(i, j) for idx, i in enumerate(pokemon_names)
         for j in pokemon_names[idx+1:]]
         
n_cols = 4
n_pairs = len(pairs)
n_rows = (n_pairs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
axes = axes.flatten()

# Plot angles for each pair
for ax, (poke1, poke2) in zip(axes, pairs):
    angle_deg = angular_sim_df.loc[poke1, poke2]
    angle_rad = np.deg2rad(angle_deg)
    
    # Unit vectors: v1 is [1, 0]; v2 is rotated by angle_rad
    v1 = np.array([1, 0])
    v2 = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='red', label=poke1)
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='blue', label=poke2)
    
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(f"{poke1} vs {poke2}\n{angle_deg:.2f}°", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8)

# Hide unused subplots (if any)
for i in range(len(pairs), len(axes)):
    fig.delaxes(axes[i])
    
plt.suptitle("Cosine Similarity (Degrees) Between Pokémon Stat Vectors", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

