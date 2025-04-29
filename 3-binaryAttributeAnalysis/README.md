# Pokémon Binary Attribute Analysis

This project calculates dissimilarity between Pokémon based on their binary attributes using the Simple Matching Coefficient (SMC).

## Features

- Binary attribute matrix construction for Pokémon characteristics
- Dissimilarity matrix calculation using SMC
- Clean tabular output display
- Customizable display options for pandas DataFrames

## Requirements

- Python 3.x
- pandas
- numpy

## Dataset Structure

The script uses a predefined dictionary of Pokémon with their binary attributes:

```python
data = {
    "Sigilyph": [0, 0, 1, 1, 0, 0, 0],
    "Togekiss": [0, 0, 1, 0, 0, 0, 0],
    "Hitmontop": [1, 1, 0, 0, 0, 0, 0],
    "Naganadel": [0, 1, 1, 0, 0, 0, 1],
    "Dragapult": [0, 1, 1, 0, 0, 0, 1],
    "Growlithe": [1, 1, 0, 0, 0, 1, 0]
}