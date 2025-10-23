import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    roc_auc_score, 
    f1_score,
    accuracy_score,
    confusion_matrix
)
import pandas as pd
import numpy as np
import json
import os
from .data_generator import NODES_FILE, TRANSACTIONS_FILE, DATA_DIR

PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def load_data():
    """Loads node and transaction data and constructs a PyG Data object."""
    try:
        nodes_df = pd.read_csv(NODES_FILE)
        transactions_df = pd.read_csv(TRANSACTIONS_FILE)
    except FileNotFoundError:
        return None

    # Node features (x) and labels (y)
    feature_cols = [col for col in nodes_df.columns if 'feature_' in col]
    x = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(nodes_df['is_fraud'].values, dtype=torch.long)

    # Edge index
    edge_index = torch.tensor(transactions_df[['sender_id', 'receiver_id']].values.T, dtype=torch.long)
    
    # Edge attributes (e.g., amount)
    edge_attr = torch.tensor(transactions_df['amount'].values, dtype=torch.float).unsqueeze(1)

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    
    # Create train/test masks (80% train, 20% test)
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_end = int(0.8 * num_nodes)
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[perm[:train_end]] = True
    
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[perm[train_end:]] = True
    
    return data

def train_and_evaluate():
    """
    Main function to load data, train the GNN, and return metrics.
    """
    data = load_data()
    if data is None:
        raise FileNotFoundError("Dataset not found. Please generate it first.")

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGEModel(
        in_channels=data.num_node_features,
        hidden_channels=32,
        out_channels=2
    ).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # Convert to numpy arrays first
        test_y = data.y[data.test_mask].cpu().numpy()
        test_pred = pred[data.test_mask].cpu().numpy()
        
        # Calculate metrics
        precision = precision_score(test_y, test_pred, zero_division=0)
        recall = recall_score(test_y, test_pred, zero_division=0)
        f1 = f1_score(test_y, test_pred, zero_division=0)
        accuracy = accuracy_score(test_y, test_pred)
        
        # For AUC, we need probabilities for the positive class
        test_probs = out[data.test_mask].exp()[:, 1].cpu().numpy()
        
        # Handle edge case where all labels are the same class
        if len(np.unique(test_y)) > 1:
            auc = roc_auc_score(test_y, test_probs)
        else:
            auc = 0.0
        
        # Confusion Matrix with robust handling
        cm = confusion_matrix(test_y, test_pred)
        
        # Handle different confusion matrix shapes
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1):
            # Only one class predicted and present
            if test_y[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            # Fallback for unexpected shapes
            tn, fp, fn, tp = 0, 0, 0, 0

    # Find Fraud Rings
    all_preds = pred.cpu().numpy()
    fraud_node_indices = np.where(all_preds == 1)[0]
    G_nx = to_networkx(data.cpu(), to_undirected=False)
    valid_fraud_nodes = [n for n in fraud_node_indices if n in G_nx]
    fraud_subgraph = G_nx.subgraph(valid_fraud_nodes)
    
    rings = list(nx.weakly_connected_components(fraud_subgraph))
    num_rings = len(rings)
    
    # Save predictions
    nodes_in_rings = set()
    for ring in rings:
        nodes_in_rings.update(ring)

    predictions_data = {
        'predictions': [int(p) for p in all_preds],  # Convert each element
        'nodes_in_rings': [int(n) for n in nodes_in_rings]  # Convert each element
    }
    
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions_data, f)
    
    # Return metrics with explicit type conversion
    return {
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "fraud_ring_count": int(num_rings),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
    }