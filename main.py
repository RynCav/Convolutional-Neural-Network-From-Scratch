from params import *
from model import *

# innilize each layer and activation function
STEPS = [ConvolutionLayer(32, 3, 1, 1, 1), ReLU(), PoolingLayer(2, 2), ConvolutionLayer(64, 3, 1, 1, 1),
         PoolingLayer(2, 2), ReLU(), FlattenLayer(BATCH_SIZE), DenseLayer(100352, 128), ReLU(), DenseLayer(128, 10),
         Softmax()]

# create the model object
model = Model(STEPS)
# train the model
model.train(Dataset, EPOCHS, BATCH_SIZE)

# test the model to determine accuracy
model.test(Dataset)

# save the model to the specified file1
save(model, 'model.pickle')
