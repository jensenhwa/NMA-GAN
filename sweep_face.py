import wandb

sweep_configuration = {
    'method': 'random',
    'name': 'sweep_with_raceff',
    'metric': {'goal': 'maximize', 'name': 'acc'},
    'parameters':
    {
        'batch_size': {'values': [32, 48]},
        # 'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 1e-3, 'min': 1e-6},
        'l': {'min': 0.0001, 'max': 1000.0},
        'g': {'min': 0.0001, 'max': 10.0},
        'r': {'min': 0.0001, 'max': 10.0},
        'partial': {'min': 1, 'max': 64}
     },
    'program': 'train_face.py',
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='nma-gan', entity='jphwa')
wandb.agent(sweep_id)
