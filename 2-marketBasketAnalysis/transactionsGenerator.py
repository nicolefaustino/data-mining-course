import csv
import random

# Define the cafe items
cafe_items = [
    "Espresso", "Americano", "Latte", "Cappuccino", "Mocha", "Macchiato",
    "Black Coffee", "Iced Coffee", "Hot Chocolate", "Tea", "Green Tea", 
    "Chai Latte", "Croissant", "Bagel", "Muffin", "Brownie", "Cheesecake", 
    "Cookie", "Donut", "Sandwich", "Pasta", "Salad", "Panini", "Wrap", 
    "Avocado Toast", "Omelette", "Granola Bowl", "Smoothie",
    "Fruit Parfait", "Waffles"
]

# Step 1: Generate a simple CSV file with random transactions
def generate_simple_csv(filename):
    transactions = []
    for _ in range(1000):  # Generate 1000 transactions
        transaction = random.sample(cafe_items, random.randint(1, len(cafe_items)))
        transactions.append(transaction)
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Transaction_ID", "Items"])
        for i, transaction in enumerate(transactions):
            writer.writerow([i+1, ",".join(transaction)])

# Generate the simple CSV file
generate_simple_csv('ishikawacafe_transactions.csv')
print("CSV file 'ishikawacafe_transactions.csv' with 1000 transactions has been generated.")