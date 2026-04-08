import os
import sys
# Fix OpenMP hang on Windows - MUST be set before any ML imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from app.trainer import MovieModelTrainer

if __name__ == '__main__':
    # Parse --quick flag for fast iteration (5 trials instead of 50)
    quick_mode = '--quick' in sys.argv
    n_trials = 5 if quick_mode else 50
    
    mode = "QUICK" if quick_mode else "FULL"
    print(f"Starting {mode} training pipeline (n_trials={n_trials})...")
    print(f"  Use --quick flag for fast 10-min test runs\n")
    
    trainer = MovieModelTrainer(n_tuning_trials=n_trials)
    results = trainer.train()
    print("\n\nTraining finished!")
    print(results)
