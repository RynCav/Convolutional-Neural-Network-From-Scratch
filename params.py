import data as d


# Hyperparameters
BATCH_SIZE = 2
EPOCHS = 10
lr = 0.001
max_lr = 0.015
dr = 0.7
DECAY = 5e-5
PATIENCE = 3

# Dataset used:
Dataset = d.Data('Datasets/Fashion MNIST/')
Dataset.normalize()
