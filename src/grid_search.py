import argparse
import itertools
import os
import json
from datetime import datetime
from trainer import ModelTrainer
from model import BirdJEPA
from data_class import BirdJEPA_Dataset
from torch.utils.data import DataLoader
import torch
from data_class import collate_fn

def create_parameter_grid():
    """Define the parameter grid for search"""
    param_grid = {
        'hidden_dim': [128, 256, 512],
        'num_layers': [4, 6, 8],
        'num_heads': [4, 8],
        'dropout': [0.1, 0.2],
        'mlp_dim': [512, 1024],          # MLP dim for encoders
        'pred_hidden_dim': [256, 384],
        'pred_num_layers': [3, 6],
        'pred_mlp_dim': [512, 1024],     # Explicit MLP dim for predictor (instead of ratio)
        'batch_size': [32, 64],
        'lr': [1e-3, 5e-4],
        'patience': [2, 4, 6],
        'ema_momentum': [0.996, 0.999]
    }
    return param_grid

def get_experiment_name(params):
    """Generate a unique experiment name based on parameters"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"grid_search_{timestamp}_h{params['hidden_dim']}_l{params['num_layers']}_head{params['num_heads']}"

def run_experiment(args, params):
    """Run a single experiment with given parameters"""
    # Create experiment directory
    experiment_name = get_experiment_name(params)
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    weights_dir = os.path.join(experiment_dir, 'saved_weights')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config = {
        **params,
        "input_dim": args.input_dim,
        "max_seq_len": args.max_seq_len,
        "mask_ratio": args.mask_ratio,
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "device": str(args.device),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Create data loaders
    dl_train = DataLoader(
        BirdJEPA_Dataset(data_dir=args.train_dir, segment_len=args.max_seq_len),
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, 
                                          segment_length=args.max_seq_len, 
                                          mask_p=args.mask_ratio)
    )

    dl_test = DataLoader(
        BirdJEPA_Dataset(data_dir=args.test_dir, segment_len=args.max_seq_len),
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, 
                                          segment_length=args.max_seq_len, 
                                          mask_p=args.mask_ratio)
    )

    # Initialize model
    model = BirdJEPA(
        input_dim=args.input_dim,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        num_heads=params['num_heads'],
        dropout=params['dropout'],
        mlp_dim=params['mlp_dim'],
        pred_hidden_dim=params['pred_hidden_dim'],
        pred_num_layers=params['pred_num_layers'],
        pred_num_heads=params['num_heads'],  # Using same as encoder
        pred_mlp_ratio=4.0,  # Fixed
        max_seq_len=args.max_seq_len
    ).to(args.device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + 
        list(model.predictor.parameters()) + 
        list(model.decoder.parameters()),
        lr=params['lr'],
        weight_decay=0.0
    )

    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=dl_train,
        test_loader=dl_test,
        optimizer=optimizer,
        device=args.device,
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        weights_save_dir=weights_dir,
        experiment_dir=experiment_dir,
        verbose=args.verbose
    )

    # Train model
    loss_history = trainer.train()
    
    # Plot results
    trainer.plot_results(save_plot=True)

    return loss_history

def main():
    parser = argparse.ArgumentParser(description='Grid Search for BirdJEPA')
    
    # Data and output directories
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=500)
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=513)
    parser.add_argument('--max_seq_len', type=int, default=500)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get parameter grid
    param_grid = create_parameter_grid()
    
    # Generate all possible combinations
    param_keys = param_grid.keys()
    param_values = param_grid.values()
    all_combinations = list(itertools.product(*param_values))
    
    # Calculate total number of experiments
    total_experiments = len(all_combinations)
    print(f"\nStarting grid search with {total_experiments} combinations")
    
    # Store results
    results = []
    
    # Run experiments
    for i, values in enumerate(all_combinations):
        params = dict(zip(param_keys, values))
        print(f"\nRunning experiment {i+1}/{total_experiments}")
        print("Parameters:", json.dumps(params, indent=2))
        
        try:
            loss_history = run_experiment(args, params)
            
            # Store results
            result = {
                'params': params,
                'final_train_loss': loss_history[0][-1] if loss_history[0] else None,
                'final_val_loss': loss_history[1][-1] if loss_history[1] else None,
                'min_val_loss': min(loss_history[1]) if loss_history[1] else None
            }
            results.append(result)
            
            # Save intermediate results
            with open(os.path.join(args.output_dir, 'grid_search_results.json'), 'w') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            print(f"Error in experiment: {str(e)}")
            continue

    # Find best model
    if results:
        best_result = min(results, key=lambda x: x['min_val_loss'])
        print("\nBest parameters:")
        print(json.dumps(best_result['params'], indent=2))
        print(f"Best validation loss: {best_result['min_val_loss']}")

if __name__ == '__main__':
    main()