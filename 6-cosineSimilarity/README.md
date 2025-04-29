# Pokémon Stat Vector Similarity Analysis

This project analyzes the similarity between Pokémon using cosine similarity on their base stats, visualizing the results both numerically and graphically.

## Features

- Computes cosine similarity between Pokémon stat vectors
- Converts similarity scores to angular degrees
- Visualizes pairwise comparisons with vector plots
- Clean tabular output of similarity matrices

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib

## Dataset Structure

The script uses a dictionary of Pokémon with their base stats:

```python
data = {
    'Pokemon': ['Sigilyph', 'Togekiss', 'Hitmontop', 'Naganadel', 'Dragapult', 'Growlithe'],
    'HP': [72, 85, 50, 73, 88, 55],
    'Attack': [58, 50, 95, 73, 120, 70],
    'Defense': [80, 95, 95, 73, 75, 45],
    'Speed': [103, 120, 35, 127, 100, 70],
    'Sp. Atk': [80, 115, 110, 73, 75, 50],
    'Sp. Def': [97, 80, 70, 121, 142, 60]
}