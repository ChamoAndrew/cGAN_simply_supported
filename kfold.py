from torch.utils.data import DataLoader, SubsetRandomSampler
from src.utils import dataset_creation
from sklearn.model_selection import KFold, train_test_split
from train import trainer
import argparse
import json
import time
import os
import numpy as np

def k_fold_runner(csv_path: str = os.path.join("kaggle", "input", "simply-supported-chamod", "Simply_supported", "Target"),
                 target_folder: str = os.path.join("kaggle", "input", "simply-supported-chamod", "Simply_supported", "Data.csv"),
                 batch_size: int = 19,
                 run_grid_search: bool = False,
                 num_folds: int = 4) -> None:
    """
    Run k-fold cross validation with optional grid search
    
    Args:
        csv_path: Path to CSV file with data information
        target_folder: Path to folder containing target images
        batch_size: Batch size for DataLoader
        run_grid_search: Whether to perform grid search
        num_folds: Number of folds for cross validation
    """
    # Create necessary directories
    os.makedirs("grid_search_results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load dataset
    Data_tg = dataset_creation(csv_path, target_folder)
    train_val_indices, _ = train_test_split(
        range(len(Data_tg)), 
        test_size=0.2, 
        random_state=42
    )
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    if run_grid_search:
        print("\n=== Starting Grid Search ===")
        
        # Define parameter grid
        param_grid = {
            'learning_rate': [0.0001, 0.0002, 0.0005],
            'lambda_l1': [10, 50, 100],
            'num_epochs': [5, 10],
            'scheduler_factor': [0.5, 0.75],
            'scheduler_patience': [3, 5]
        }
        
        best_params = None
        best_val_loss = float('inf')
        all_results = []
        
        # Generate all parameter combinations
        from itertools import product
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, combo)) for combo in product(*values)]
        
        for params in combinations:
            print(f"\nTesting combination {len(all_results)+1}/{len(combinations)}:")
            print(json.dumps(params, indent=4))
            
            fold_val_losses = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
                print(f"  Fold {fold+1}/{num_folds}")
                
                train_sampler = SubsetRandomSampler([train_val_indices[i] for i in train_idx])
                val_sampler = SubsetRandomSampler([train_val_indices[i] for i in val_idx])

                train_loader = DataLoader(
                    Data_tg, 
                    batch_size=batch_size, 
                    sampler=train_sampler, 
                    pin_memory=True,
                    num_workers=4
                )
                val_loader = DataLoader(
                    Data_tg, 
                    batch_size=batch_size, 
                    sampler=val_sampler, 
                    pin_memory=True,
                    num_workers=4
                )
                
                # Run training with current parameters
                train_hist = trainer(
                    train_loader,
                    val_loader,
                    fold,
                    experiment_name=f"grid_search/comb_{len(all_results)+1}_fold_{fold+1}",
                    **params
                )
                
                fold_val_losses.append(min(train_hist['val_losses']))
            
            # Calculate average validation loss
            avg_val_loss = np.mean(fold_val_losses)
            std_val_loss = np.std(fold_val_losses)
            
            print(f"  Avg Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
            
            all_results.append({
                'params': params,
                'avg_val_loss': avg_val_loss,
                'std_val_loss': std_val_loss,
                'fold_val_losses': fold_val_losses
            })
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_params = params
                print(f"  New best combination found!")
        
        # Save grid search results
        timestamp = int(time.time())
        results_file = os.path.join("grid_search_results", f"grid_search_results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump({
                'best_params': best_params,
                'best_val_loss': best_val_loss,
                'all_results': all_results,
                'param_grid': param_grid
            }, f, indent=4)
        
        print(f"\nSaved grid search results to {results_file}")
        print("\n=== Starting Full Training with Best Parameters ===")
        
        # Reset KFold for final training
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Run final k-fold training (with best params if grid search was done)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
        print(f"\n--- Fold {fold+1}/{num_folds} ---")
        
        train_sampler = SubsetRandomSampler([train_val_indices[i] for i in train_idx])
        val_sampler = SubsetRandomSampler([train_val_indices[i] for i in val_idx])

        train_loader = DataLoader(
            Data_tg, 
            batch_size=batch_size, 
            sampler=train_sampler, 
            pin_memory=True,
            num_workers=4
        )
        val_loader = DataLoader(
            Data_tg, 
            batch_size=batch_size, 
            sampler=val_sampler, 
            pin_memory=True,
            num_workers=4
        )
        
        if run_grid_search:
            trainer(
                train_loader, 
                val_loader, 
                fold,
                experiment_name=f"best_params_fold_{fold+1}",
                **best_params
            )
        else:
            trainer(
                train_loader, 
                val_loader, 
                fold,
                experiment_name=f"default_fold_{fold+1}"
            )

if __name__ == "__main__":
    DEFAULT_CSV_PATH = os.path.join("Simply_supported", "Data.csv")
    DEFAULT_TARGET_FOLDER = os.path.join("Simply_supported", "Target")
    
    parser = argparse.ArgumentParser(
        description="Run k-fold cross validation for diffusion GAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--target_folder", type=str, default=DEFAULT_TARGET_FOLDER)
    parser.add_argument("--batch_size", type=int, default=19)
    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--num_folds", type=int, default=4)
    
    args = parser.parse_args()
    
    print("\n=== Configuration ===")
    print(f"CSV Path: {args.csv_path}")
    print(f"Target Folder: {args.target_folder}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Grid Search: {args.grid_search}")
    print(f"Number of Folds: {args.num_folds}")
    print("===================\n")
    
    k_fold_runner(
        csv_path=args.csv_path,
        target_folder=args.target_folder,
        batch_size=args.batch_size,
        run_grid_search=args.grid_search,
        num_folds=args.num_folds
    )