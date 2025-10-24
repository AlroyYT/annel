import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random  # <-- 1. ADD THIS IMPORT

# Configuration
# We are moving these inside the function, so they can be removed from here.
# NUM_NODES = 1000
# NUM_TRANSACTIONS = 5000
# FRAUD_NODE_PERCENTAGE = 0.05
NUM_FEATURES = 16

# Create a directory for data if it doesn't exist
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

NODES_FILE = os.path.join(DATA_DIR, "nodes.csv")
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.csv")

def generate_synthetic_data():
    """
    Generates and saves a synthetic dataset of nodes and transactions.
    """
    
    # --- vvv 2. ADD THIS RANDOMIZED CONFIG vvv ---
    NUM_NODES = random.randint(900, 1500)
    NUM_TRANSACTIONS = random.randint(4000, 7000)
    FRAUD_NODE_PERCENTAGE = random.uniform(0.04, 0.10)  # 4% to 10%
    # --- ^^^ END OF NEW CODE ^^^ ---

    # 1. Generate Nodes
    num_fraud_nodes = int(NUM_NODES * FRAUD_NODE_PERCENTAGE)
    num_normal_nodes = NUM_NODES - num_fraud_nodes
    
    node_ids = np.arange(NUM_NODES)
    is_fraud = np.zeros(NUM_NODES, dtype=int)
    fraud_node_indices = np.random.choice(node_ids, num_fraud_nodes, replace=False)
    is_fraud[fraud_node_indices] = 1
    
    # Generate random features
    features = np.random.rand(NUM_NODES, NUM_FEATURES)
    
    # Inject a signal for fraud nodes
    # Let's make the first 3 features slightly higher for fraud nodes
    if num_fraud_nodes > 0: # Add a check in case 0 fraud nodes are generated
        features[fraud_node_indices, 0:3] += np.random.uniform(0.5, 1.0, (num_fraud_nodes, 3))
    
    nodes_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(NUM_FEATURES)])
    nodes_df['node_id'] = node_ids
    nodes_df['is_fraud'] = is_fraud
    
    # 2. Generate Transactions (Edges)
    senders = []
    receivers = []
    amounts = []
    timestamps = []
    is_fraud_transaction = []
    
    normal_node_indices = np.where(is_fraud == 0)[0]
    
    # Create fraud rings (dense subgraphs)
    num_fraud_rings = 5
    # Handle case where num_fraud_nodes is too small
    if num_fraud_nodes < num_fraud_rings * 2:
        num_fraud_rings = 1
        ring_size = num_fraud_nodes
    else:
        ring_size = num_fraud_nodes // num_fraud_rings
        
    num_ring_transactions = int(NUM_TRANSACTIONS * 0.2)  # 20% of transactions are in rings
    
    for i in range(num_fraud_rings):
        ring_nodes = fraud_node_indices[i*ring_size : (i+1)*ring_size]
        if len(ring_nodes) < 2:
            continue
            
        for _ in range(num_ring_transactions // num_fraud_rings):
            sender, receiver = np.random.choice(ring_nodes, 2, replace=False)
            senders.append(sender)
            receivers.append(receiver)
            amounts.append(np.random.uniform(500, 2000)) # Higher value transactions
            timestamps.append(datetime.now() - timedelta(days=np.random.randint(0, 30)))
            is_fraud_transaction.append(1)

    # Create 'laundering' transactions (fraud -> normal)
    num_laundering_transactions = int(NUM_TRANSACTIONS * 0.1)
    if num_fraud_nodes > 0 and len(normal_node_indices) > 0: # Check nodes exist
        for _ in range(num_laundering_transactions):
            senders.append(np.random.choice(fraud_node_indices))
            receivers.append(np.random.choice(normal_node_indices))
            amounts.append(np.random.uniform(100, 1000))
            timestamps.append(datetime.now() - timedelta(days=np.random.randint(0, 30)))
            is_fraud_transaction.append(1) # Transaction is part of fraud

    # Create normal transactions
    num_normal_transactions = NUM_TRANSACTIONS - len(senders)
    if len(normal_node_indices) > 1: # Check at least 2 normal nodes exist
        for _ in range(num_normal_transactions):
            sender, receiver = np.random.choice(normal_node_indices, 2, replace=False)
            senders.append(sender)
            receivers.append(receiver)
            amounts.append(np.random.uniform(10, 200)) # Lower value transactions
            timestamps.append(datetime.now() - timedelta(days=np.random.randint(0, 30)))
            is_fraud_transaction.append(0)
        
    transactions_df = pd.DataFrame({
        'sender_id': senders,
        'receiver_id': receivers,
        'amount': amounts,
        'timestamp': timestamps,
        'is_fraud_transaction': is_fraud_transaction
    })
    
    # Save to CSV
    nodes_df.to_csv(NODES_FILE, index=False)
    transactions_df.to_csv(TRANSACTIONS_FILE, index=False)
    
    return {
        "nodes": NUM_NODES,
        "transactions": len(transactions_df), # Use the actual number of transactions created
        "fraudulent_nodes": num_fraud_nodes
    }

if __name__ == "__main__":
    stats = generate_synthetic_data()
    print(f"Data generated: {stats}")