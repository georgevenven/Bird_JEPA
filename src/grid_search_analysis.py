# analyze_grid_search.py

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description='deep analysis for jepa grid search results, going full david goggins on these combos.'
    )
    parser.add_argument('--results_file', type=str, required=True,
                        help='path to the "grid_search_results.json" file generated by the search.')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                        help='directory to store analysis plots.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of top combos to print in detail.')
    parser.add_argument('--bottom_k', type=int, default=5,
                        help='number of worst combos to print in detail.')
    parser.add_argument('--plot', action='store_true',
                        help='if set, produce some summary plots.')
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load the results
    with open(args.results_file, 'r') as f:
        results = json.load(f)

    # sanity check
    if not results:
        print("no results found in the results_file.")
        return

    # results is a list of dicts. each dict has:
    # {
    #    'params': {dictionary of hyperparams}, 
    #    'final_train_loss': float or None,
    #    'final_val_loss': float or None,
    #    'min_val_loss': float or None,
    #    'avg_val_loss': float or None,
    #    'convergence_step': int
    # }
    # we are going to do a thorough analysis

    # sort by min_val_loss ascending
    sorted_results = sorted(results, key=lambda x: x.get('min_val_loss', float('inf')))

    # print top k combos
    print("\n=== top performing combos (lowest min_val_loss) ===")
    top_k = min(args.top_k, len(sorted_results))
    for i in range(top_k):
        r = sorted_results[i]
        print(f"\nrank {i+1}/{top_k}: min_val_loss={r['min_val_loss']:.4f}")
        print("parameters:")
        for k, v in r['params'].items():
            print(f"  {k} = {v}")
        print(f"final_train_loss: {r['final_train_loss']}, final_val_loss: {r['final_val_loss']}, "
              f"avg_val_loss: {r['avg_val_loss']:.4f}, convergence_step: {r['convergence_step']}")

    # print bottom k combos
    print("\n=== worst performing combos (highest min_val_loss) ===")
    bottom_k = min(args.bottom_k, len(sorted_results))
    for i in range(bottom_k):
        r = sorted_results[-(i+1)]
        print(f"\nrank -{i+1}/{bottom_k}: min_val_loss={r['min_val_loss']:.4f}")
        print("parameters:")
        for k, v in r['params'].items():
            print(f"  {k} = {v}")
        print(f"final_train_loss: {r['final_train_loss']}, final_val_loss: {r['final_val_loss']}, "
              f"avg_val_loss: {r['avg_val_loss']:.4f}, convergence_step: {r['convergence_step']}")

    # basic param correlation analysis
    # we can attempt to see if any param strongly correlates with min_val_loss
    # note that many are categorical or discrete, so we do a simplistic approach
    # let's gather all param -> list of (value, min_val_loss)
    param_to_data = defaultdict(list)
    for r in results:
        min_loss = r.get('min_val_loss', None)
        if min_loss is None:
            continue
        for k, v in r['params'].items():
            param_to_data[k].append((v, min_loss))

    # now we compute correlation for continuous numeric params
    # we'll skip non-numeric ones or do a naive approach
    print("\n=== param correlations vs. min_val_loss (naive approach) ===")
    for param, pairs in param_to_data.items():
        # see if v is numeric. we only do correlation if 10+ distinct numeric
        # a simpler approach is just to see if param is float or int
        # gather all numeric pairs
        numeric_values = []
        numeric_losses = []
        for (val, loss) in pairs:
            if isinstance(val, (int, float)):
                numeric_values.append(val)
                numeric_losses.append(loss)
        if len(numeric_values) < 2:
            continue
        # attempt correlation
        # (we skip if there's not enough variety in numeric_values)
        if len(set(numeric_values)) > 1:
            corr = np.corrcoef(numeric_values, numeric_losses)[0,1]
            print(f"param '{param}': correlation = {corr:.3f} (n={len(numeric_values)})")

    # if user wants to produce plots
    if args.plot:
        # 1) plot min_val_loss distribution
        min_losses = [r['min_val_loss'] for r in sorted_results if r['min_val_loss'] is not None]
        plt.figure(figsize=(10,6))
        plt.hist(min_losses, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title('distribution of min_val_loss across all combos')
        plt.xlabel('min_val_loss')
        plt.ylabel('count')
        plt.grid(True)
        hist_out = os.path.join(args.output_dir, 'min_val_loss_hist.png')
        plt.savefig(hist_out)
        plt.close()
        print(f"\nplotted distribution of min_val_loss to {hist_out}")

        # 2) let's pick two numeric parameters to do a scatter plot vs. min_val_loss
        #    for example 'encoder_lr' and 'predictor_lr'
        #    if they exist in param_to_data
        if 'encoder_lr' in param_to_data and 'predictor_lr' in param_to_data:
            # gather data
            # we need to find the same set of combos that have both param
            combos = []
            for r in sorted_results:
                param_dict = r['params']
                if 'encoder_lr' in param_dict and 'predictor_lr' in param_dict and r.get('min_val_loss') is not None:
                    combos.append( (param_dict['encoder_lr'], param_dict['predictor_lr'], r['min_val_loss']) )
            if combos:
                x = [c[0] for c in combos]
                y = [c[1] for c in combos]
                z = [c[2] for c in combos]
                # scatter plot of x vs y, color by z
                plt.figure(figsize=(8,6))
                sc = plt.scatter(x, y, c=z, cmap='viridis', edgecolor='black')
                plt.colorbar(sc, label='min_val_loss')
                plt.xlabel('encoder_lr')
                plt.ylabel('predictor_lr')
                plt.title('encoder_lr vs predictor_lr vs min_val_loss')
                scatter_out = os.path.join(args.output_dir, 'enc_lr_vs_pred_lr_scatter.png')
                plt.savefig(scatter_out)
                plt.close()
                print(f"plotted encoder_lr vs predictor_lr vs min_val_loss to {scatter_out}")

    # final summary
    best = sorted_results[0]
    print("\n=== best overall combination ===")
    print(f"min_val_loss = {best['min_val_loss']:.4f}")
    print("parameters:")
    for k, v in best['params'].items():
        print(f"  {k} = {v}")
    print(f"final_train_loss: {best['final_train_loss']}, final_val_loss: {best['final_val_loss']}, "
          f"avg_val_loss: {best['avg_val_loss']:.4f}, convergence_step: {best['convergence_step']}")

if __name__ == '__main__':
    main()
