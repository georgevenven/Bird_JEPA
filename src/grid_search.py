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
        'input_dim': [513],
        'max_seq_len': [500],
        'mask_ratio': [0.3],
        'hidden_dim': [128],
        'num_layers': [2],
        'num_heads': [2],
        'dropout': [0.0],
        'mlp_dim': [256],
        'pred_hidden_dim': [128],
        'pred_num_layers': [2],
        'pred_mlp_dim': [256],
        'batch_size': [16],
        
        'encoder_lr': [1e-4],
        'predictor_lr': [1e-4],
        'decoder_lr': [1e-4],
        
        'freeze_encoder_steps': [0, 1000],
        'freeze_decoder_steps': [0, 1000],
        
        'patience': [4],
        'ema_momentum': [0.9, 0.99],
        'steps': [20000],
        'eval_interval': [500]
    }
    return param_grid

def get_experiment_name(params):
    """Generate a unique experiment name based on parameters"""
    timestamp = datetime.now().strftime("%m%d_%H%M")  # Shorter timestamp format
    
    # Create compact parameter string
    model_params = f"h{params['hidden_dim']}_l{params['num_layers']}_d{str(params['dropout']).replace('0.','')}"
    training_params = f"bs{params['batch_size']}_elr{str(params['encoder_lr']).replace('0.','').replace('-','')}_plr{str(params['predictor_lr']).replace('0.','').replace('-','')}_dlr{str(params['decoder_lr']).replace('0.','').replace('-','')}"
    ema = f"ema{str(params['ema_momentum']).replace('0.','')}"
    
    return f"jepa_{model_params}_{training_params}_{ema}_{timestamp}"

def run_experiment(args, params):
    """Run a single experiment with given parameters"""
    # Create experiment directory
    experiment_name = get_experiment_name(params)
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    weights_dir = os.path.join(experiment_dir, 'saved_weights')
    os.makedirs(experiment_dir, exist_ok=True)

    # Extract training control parameters AFTER creating experiment name
    steps = params.pop('steps')
    eval_interval = params.pop('eval_interval')
    patience = params.pop('patience')
    ema_momentum = params.pop('ema_momentum')
    
    # Extract learning rates and freeze steps
    encoder_lr = params.pop('encoder_lr')
    predictor_lr = params.pop('predictor_lr')
    decoder_lr = params.pop('decoder_lr')
    freeze_encoder_steps = params.pop('freeze_encoder_steps')
    freeze_decoder_steps = params.pop('freeze_decoder_steps')
    
    # Save full configuration with all parameters
    config = {
        **params,
        "steps": steps,
        "eval_interval": eval_interval,
        "patience": patience,
        "ema_momentum": ema_momentum,
        "encoder_lr": encoder_lr,
        "predictor_lr": predictor_lr,
        "decoder_lr": decoder_lr,
        "freeze_encoder_steps": freeze_encoder_steps,
        "freeze_decoder_steps": freeze_decoder_steps,
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "device": str(args.device),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Create data loaders using params
    dl_train = DataLoader(
        BirdJEPA_Dataset(data_dir=args.train_dir, segment_len=params['max_seq_len']),
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, 
                                          segment_length=params['max_seq_len'], 
                                          mask_p=params['mask_ratio'])
    )

    dl_test = DataLoader(
        BirdJEPA_Dataset(data_dir=args.test_dir, segment_len=params['max_seq_len']),
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, 
                                          segment_length=params['max_seq_len'], 
                                          mask_p=params['mask_ratio'])
    )

    # Initialize model using params
    model = BirdJEPA(
        input_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        num_heads=params['num_heads'],
        dropout=params['dropout'],
        mlp_dim=params['mlp_dim'],
        pred_hidden_dim=params['pred_hidden_dim'],
        pred_num_layers=params['pred_num_layers'],
        pred_num_heads=params['num_heads'],
        pred_mlp_dim=params['pred_mlp_dim'],
        max_seq_len=params['max_seq_len']
    ).to(args.device)

    # Initialize trainer with separate learning rates and freeze steps
    trainer = ModelTrainer(
        model=model,
        train_loader=dl_train,
        test_loader=dl_test,
        encoder_lr=encoder_lr,
        predictor_lr=predictor_lr,
        decoder_lr=decoder_lr,
        freeze_encoder_steps=freeze_encoder_steps,
        freeze_decoder_steps=freeze_decoder_steps,
        device=args.device,
        max_steps=steps,
        eval_interval=eval_interval,
        save_interval=args.save_interval,
        weights_save_dir=weights_dir,
        experiment_dir=experiment_dir,
        early_stopping=True,
        patience=patience,
        ema_momentum=ema_momentum,
        verbose=args.verbose
    )

    # Train model and get loss history
    train_losses, val_losses = trainer.train()
    
    # Calculate evaluation metrics
    eval_metrics = {
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'min_val_loss': min(val_losses) if val_losses else None,
        'avg_val_loss': sum(val_losses) / len(val_losses) if val_losses else None,
        'convergence_step': len(train_losses)  # Number of steps before stopping
    }
    
    # Plot and save results
    trainer.plot_results(save_plot=True)

    return eval_metrics

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
    
    # Additional parameters
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--predictor_lr', type=float, default=1e-3)
    parser.add_argument('--decoder_lr', type=float, default=1e-4)
    parser.add_argument('--freeze_encoder_steps', type=int, default=0)
    parser.add_argument('--freeze_decoder_steps', type=int, default=0)
    
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
            eval_metrics = run_experiment(args, params)
            
            # Store results
            result = {
                'params': params,
                'final_train_loss': eval_metrics['final_train_loss'],
                'final_val_loss': eval_metrics['final_val_loss'],
                'min_val_loss': eval_metrics['min_val_loss'],
                'avg_val_loss': eval_metrics['avg_val_loss'],
                'convergence_step': eval_metrics['convergence_step']
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


# python src/grid_search.py --train_dir /media/george-vengrovski/George-SSD/llb_stuff/combined_train --test_dir /media/george-vengrovski/George-SSD/llb_stuff/combined_test --output_dir ./grid_search_results --verbose
