from utils import *
from model import *
from params import *

# innilize each layer and activation function
STEPS = [ConvolutionLayer(32, 3, 3, 1, 1, 1), BatchNormalization(), ReLU(),
         PoolingLayer(2, 2, 2),
         ConvolutionLayer(64, 3, 3, 32, 1, 1), BatchNormalization(), ReLU(),
         PoolingLayer(2, 2,2),
         FlattenLayer(),
         DenseLayer(3136, 1000), BatchNormalization(), ReLU(), DropoutLayer(),
         DenseLayer(1000, 500), BatchNormalization(), ReLU(), DropoutLayer(),
         DenseLayer(500, 10), Softmax()]

# create the model object
Model = Model(STEPS, dataset=Dataset)

# train the model
Model.train(EPOCHS, BATCH_SIZE)

# test the model to determine accuracy
Model.test()

# save the model to the specified file1
Model.save_model()
