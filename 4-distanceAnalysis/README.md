# Pokémon Statistical Distance Analysis

This project calculates and compares distances between Pokémon based on their core battle statistics using three different distance metrics.

## Features

- Numerical attribute matrix of Pokémon stats
- Three distance metric calculations:
  - Manhattan (City Block) Distance
  - Euclidean Distance
  - Supremum (Chebyshev) Distance
- Clean tabular output display

## Requirements

- Python 3.x
- numpy
- pandas

## Dataset Structure

The script uses a predefined matrix of Pokémon base stats:

```python
stats_matrix = np.array([
    [72, 58, 80, 103, 80, 97],    # Sigilyph
    [85, 50, 95, 120, 115, 80],    # Togekiss
    [50, 95, 95, 35, 110, 70],     # Hitmontop
    [73, 73, 73, 127, 73, 121],    # Naganadel
    [88, 120, 75, 100, 75, 142],   # Dragapult
    [55, 70, 45, 70, 50, 60]       # Growlithe
])