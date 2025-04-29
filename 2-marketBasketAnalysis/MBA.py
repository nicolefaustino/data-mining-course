import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the transactions from a CSV file
print("Please enter the file name of the CSV file (e.g., 'pokemon_transactions.csv'): ishikawacafe_transactions.csv")
df = pd.read_csv('ishikawacafe_transactions.csv')

# Print the transaxtions
print("\n====== ISHIKAWA CAFE ======")
print("My Transaction Dataset\n")
print(df.to_string(index=False))

# Convert the transactions to a one-hot encoded format
one_hot = df['Items'].str.get_dummies(sep=',')

# Ensure the DataFrame contains boolean values
one_hot = one_hot.astype(bool)

# Generate frequent itemsets with a minimum support threshold 0f 20%
frequent_itemsets = apriori(one_hot, min_support=0.2, use_colnames=True)
print("\n=== Frequent Itemsets ===")
print(frequent_itemsets)

# Generate association rules from the frequent itemsets using a minimum confidence threshold of 50%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print("\n=== Association Rules ===")
# Display the rule along with its support, confidence, and lift
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])