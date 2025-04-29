# Pokémon Mixed Attribute Distance Analysis

This project calculates distance matrices between Pokémon using different attribute types (nominal, numerical, ordinal, and binary) and combines them into an overall distance measure.

## Features

- Handles four different attribute types:
  - Nominal (Type)
  - Numerical (Total Stats)
  - Ordinal (Evolution Stage)
  - Binary (Legendary Status)
- Calculates distance matrices for each attribute type
- Computes an overall combined distance matrix
- Clean tabular output display

## Requirements

- Python 3.x
- numpy
- pandas

## Dataset Structure

The script uses a DataFrame with the following attributes:

```python
pokemon_data = pd.DataFrame({
    "Name": ["Sigilyph", "Togekiss", "Hitmontop", "Naganadel", "Dragapult", "Growlithe"],
    "Type": ["Psychic & Flying", "Fairy & Flying", "Fighting", 
             "Poison & Dragon", "Dragon & Ghost", "Fire"],  # Nominal
    "Total Stats": [490, 545, 455, 540, 600, 350],  # Numerical
    "Evolution Stage": [1, 3, 2, 2, 3, 1],  # Ordinal
    "Legendary": [0, 0, 0, 1, 0, 0]  # Binary
})