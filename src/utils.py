import torch
import json
import os
from models import BirdJEPA
import glob
import pandas as pd
from pathlib import Path
from models.birdjepa import BirdJEPA, BJConfig
import torch.nn as nn


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_weights(dir, model):
    """Load weights from a checkpoint file"""
    checkpoint = torch.load(dir, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_model(experiment_folder, return_checkpoint=False, load_weights=True, random_weights=False):
    """
    Load a trained BirdJEPA model from an experiment folder or initialize with random weights.
    
    args:
        experiment_folder (str): path to the experiment folder containing saved weights
        return_checkpoint (bool): if true, returns full checkpoint dict for continued training
        load_weights (bool): if true, loads weights from the checkpoint file
        random_weights (bool): if true, initializes model with random weights instead of loading
    returns:
        model: loaded BirdJEPA model with weights
        checkpoint (optional): full checkpoint dict if return_checkpoint=true
    """
    config_path = os.path.join(experiment_folder, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Parse the architecture string to create blocks_config
    blocks_config = []
    if 'architecture' in config:
        for block_spec in config['architecture'].split(','):
            parts = block_spec.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid block specification: {block_spec}")
            
            block_type, param = parts
            if block_type.lower() == "local":
                blocks_config.append({"type": "local", "window_size": int(param)})
            elif block_type.lower() == "global":
                blocks_config.append({"type": "global", "stride": int(param)})
            else:
                raise ValueError(f"Unknown block type: {block_type}")
    
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
        max_seq_len=config['max_seq_len'],
        blocks_config=blocks_config  # Pass the parsed blocks_config
    )
    
    if not random_weights:
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
        
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\nLoaded checkpoint details:")
        print(f"Path: {latest_checkpoint}")
        print(f"Training step: {checkpoint['step']}")
        if 'encoder_optimizer_state' in checkpoint:
            print(f"Encoder optimizer state included: Yes")
        if 'predictor_optimizer_state' in checkpoint:
            print(f"Predictor optimizer state included: Yes")
        if 'decoder_optimizer_state' in checkpoint:
            print(f"Decoder optimizer state included: Yes")
        
        if return_checkpoint:
            return model, checkpoint, config
    else:
        print("Initialized model with random weights.")
    
    return model

def build_label_map(csv_path: str):
    df = pd.read_csv(csv_path, usecols=['filename', 'primary_label'])
    df['filename'] = df['filename'].apply(lambda s: Path(s).name)
    fname2lab = dict(zip(df.filename, df.primary_label))
    classes   = sorted(df.primary_label.unique())
    return fname2lab, classes

class _StemSeq(nn.Module):
    def __init__(self, stem, proj):
        super().__init__()
        self.stem, self.proj = stem, proj
    def forward(self, x):             # x (B,1,F,T)
        z = self.stem(x)              # (B,C,F',T')
        z = z.permute(0,3,1,2).flatten(2)  # (B,T',C*F')
        return self.proj(z)           # (B,T',D)

def load_pretrained_encoder(cfg: BJConfig, ckpt_path: str | None):
    # if no checkpoint → random init
    if not ckpt_path or str(ckpt_path).lower() in {"none", "null"}:
        model = BirdJEPA(cfg)
        return nn.Sequential(
            _StemSeq(model.stem, model.proj),
            model.encoder
        )

    sd = torch.load(ckpt_path, map_location="cpu")
    model = BirdJEPA(cfg)
    model.load_state_dict(sd, strict=False)
    # keep the projector even when we load weights
    return nn.Sequential(_StemSeq(model.stem, model.proj), model.encoder)
