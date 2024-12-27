
# ~~~~ import torch
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
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_model(experiment_folder, return_checkpoint=False):
    """
    load a trained BirdJEPA model from an experiment folder.
    
    args:
        experiment_folder (str): path to the experiment folder containing saved weights
        return_checkpoint (bool): if true, returns full checkpoint dict for continued training
        
    returns:
        model: loaded BirdJEPA model with weights
        checkpoint (optional): full checkpoint dict if return_checkpoint=true
    """
    config_path = os.path.join(experiment_folder, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = BirdJEPA(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        mlp_dim=config['mlp_dim'],
        pred_hidden_dim=config['pred_hidden_dim'],
        pred_num_layers=config['pred_num_layers'],
        pred_num_heads=config.get('pred_num_heads', config['num_heads']),
        pred_mlp_dim=config['pred_mlp_dim'],
        max_seq_len=config['max_seq_len']
    )
    
    weights_dir = os.path.join(experiment_folder, 'saved_weights')
    checkpoints = glob.glob(os.path.join(weights_dir, 'checkpoint_*.pt'))
    if not checkpoints:
        print(f"Available files in {experiment_folder}:")
        print("\n".join(os.listdir(experiment_folder)))
        print(f"\nAvailable files in weights dir {weights_dir} (if exists):")
        if os.path.exists(weights_dir):
            print("\n".join(os.listdir(weights_dir)))
        raise ValueError(f"No checkpoints found in {weights_dir}")
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nLoaded checkpoint details:")
    print(f"Path: {latest_checkpoint}")
    print(f"Training step: {checkpoint['step']}")
    print(f"Model state keys: {list(checkpoint['model_state_dict'].keys())}")
    if 'encoder_optimizer_state' in checkpoint:
        print(f"Encoder optimizer state included: Yes")
    if 'predictor_optimizer_state' in checkpoint:
        print(f"Predictor optimizer state included: Yes")
    if 'decoder_optimizer_state' in checkpoint:
        print(f"Decoder optimizer state included: Yes")
    
    if return_checkpoint:
        return model, checkpoint, config
    return model
