import torch
import json
import os
from model import BirdJEPA
import glob

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_weights(dir, model):
    """Load weights from a checkpoint file"""
    checkpoint = torch.load(dir, map_location='cpu')
    # Extract just the model state dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_model(experiment_folder):
    """Load model from experiment folder"""
    # Load config
    config_path = os.path.join(experiment_folder, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model with config parameters
    model = BirdJEPA(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        mlp_dim=config['mlp_dim'],
        pred_hidden_dim=config['pred_hidden_dim'],
        pred_num_layers=config['pred_num_layers'],
        pred_num_heads=config['pred_num_heads'],
        pred_mlp_ratio=config['pred_mlp_ratio'],
        max_seq_len=config['max_seq_len']
    )
    
    # Find latest checkpoint
    weight_path = os.path.join(experiment_folder, 'saved_weights')
    checkpoints = glob.glob(os.path.join(weight_path, 'checkpoint_*.pt'))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {weight_path}")
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Load weights
    load_weights(dir=latest_checkpoint, model=model)
    
    return model
