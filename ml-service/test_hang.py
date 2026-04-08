import sys
print("starting", flush=True)
try:
    print("step 1", flush=True)
    from app.trainer import MovieModelTrainer
    print("step 2", flush=True)
    trainer = MovieModelTrainer(n_tuning_trials=2)
    print("step 3", flush=True)
    trainer.train()
    print("step 4", flush=True)
except Exception as e:
    print(e)
