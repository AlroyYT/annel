from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import json  # <-- ADD THIS IMPORT
from . import data_generator
from . import gnn_model
from .gnn_model import PREDICTIONS_FILE # <-- Import the file path
import os

app = FastAPI(
    title="Fraud Ring Detection API",
    description="Uses a GNN to detect sophisticated fraud rings.",
)

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate_dataset")
async def api_generate_dataset():
    """
    Endpoint to generate a new synthetic dataset.
    """
    try:
        stats = data_generator.generate_synthetic_data()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_gnn")
async def api_train_gnn():
    """
    Endpoint to train the GNN model on the existing dataset.
    """
    if not os.path.exists(data_generator.NODES_FILE) or not os.path.exists(data_generator.TRANSACTIONS_FILE):
        raise HTTPException(
            status_code=400, 
            detail="Dataset not found. Please generate the dataset first via POST /generate_dataset"
        )
        
    try:
        metrics = gnn_model.train_and_evaluate()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.get("/get_dataset")
async def api_get_dataset():
    """
    Endpoint to retrieve the generated dataset as JSON.
    """
    if not os.path.exists(data_generator.NODES_FILE) or not os.path.exists(data_generator.TRANSACTIONS_FILE):
        raise HTTPException(
            status_code=400, 
            detail="Dataset not found. Please generate it first."
        )
    
    try:
        nodes_df = pd.read_csv(data_generator.NODES_FILE)
        transactions_df = pd.read_csv(data_generator.TRANSACTIONS_FILE)
        
        # Round floats for cleaner JSON
        nodes_df = nodes_df.round(4)
        transactions_df['amount'] = transactions_df['amount'].round(2)
        
        return {
            "nodes": nodes_df.to_dict('records'),
            "transactions": transactions_df.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- vvv NEW ENDPOINT vvv ---

@app.get("/get_graph_data")
async def api_get_graph_data():
    """
    Endpoint to retrieve data formatted for graph visualization.
    Needs to be called AFTER /train_gnn.
    """
    files_exist = (
        os.path.exists(data_generator.NODES_FILE) and
        os.path.exists(data_generator.TRANSACTIONS_FILE) and
        os.path.exists(PREDICTIONS_FILE)
    )
    if not files_exist:
        raise HTTPException(
            status_code=400, 
            detail="Required data not found. Please generate dataset and train model first."
        )

    try:
        nodes_df = pd.read_csv(data_generator.NODES_FILE)
        transactions_df = pd.read_csv(data_generator.TRANSACTIONS_FILE)
        
        with open(PREDICTIONS_FILE, 'r') as f:
            predictions_data = json.load(f)
        
        preds = predictions_data['predictions']
        ring_nodes = set(predictions_data['nodes_in_rings'])
        
        # Build node list for the graph
        graph_nodes = []
        for _, row in nodes_df.iterrows():
            node_id = int(row['node_id'])
            features = {k: v for k, v in row.items() if k not in ['node_id', 'is_fraud']}
            
            graph_nodes.append({
                "id": node_id,
                "is_fraud_actual": int(row['is_fraud']),
                "is_fraud_predicted": preds[node_id],
                "is_in_ring": node_id in ring_nodes,
                "features": features
            })
            
        # Build link list for the graph
        graph_links = []
        for _, row in transactions_df.iterrows():
            graph_links.append({
                "source": int(row['sender_id']),
                "target": int(row['receiver_id']),
                "amount": round(row['amount'], 2),
                "is_fraud": int(row['is_fraud_transaction'])
            })
            
        return {
            "nodes": graph_nodes,
            "links": graph_links
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing graph data: {str(e)}")

# --- ^^^ NEW ENDPOINT ^^^ ---

if __name__ == "__main__":
    print("This file is not meant to be run directly.")
    print("Run from the 'backend' directory using: uvicorn app.main:app --reload --port 8000")