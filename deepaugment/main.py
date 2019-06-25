from deepaugment import DeepAugment

my_config = {
    "model": "basiccnn",
    "method": "bayesian_optimization",
    "train_set_size": 2000,
    "opt_samples": 1,
    "opt_last_n_epochs": 3,
    "opt_initial_points": 1,
    "child_epochs": 1,
    "child_first_train_epochs": 0,
    "child_batch_size": 64,
    "notebook_path": '/content/results' 
    
}

# X_train.shape -> (N, M, M, 3)
# y_train.shape -> (N)
deepaug = DeepAugment(images=x_train, labels=y_train, config=my_config)
