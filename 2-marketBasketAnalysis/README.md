# Market Basket Analysis for Ishikawa Cafe

This project performs market basket analysis on transaction data from Ishikawa Cafe using the Apriori algorithm to discover association rules between items.

## Features

- Transaction data processing from CSV files
- One-hot encoding of transaction items
- Frequent itemset generation with configurable minimum support
- Association rule mining with confidence metrics
- Clear display of discovered patterns and rules

## Requirements

- Python 3.x
- pandas
- mlxtend

## Dataset Format

The CSV file should contain transaction data with:
- One column named "Items"
- Items in each transaction separated by commas