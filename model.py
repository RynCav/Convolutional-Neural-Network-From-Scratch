import copy
import math
import numpy as np
import pickle as pkl
import scipy


"""Helper functions for initializing weights"""
# for ReLU
def he(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])


# for other
def xavier(shape):
    return np.random.randn(*shape) * np.sqrt(1 / shape[0])


# for random
def normal(shape):
    return np.random.uniform(0, 0.1, shape)


# a single dense layer of a neural network
class DenseLayer:
    def __init__(self, n_inputs, n_neurons, innitlized = he, l1=0, l2=0):
        self.weights = innitlized((n_inputs, n_neurons))

        # set all biases in array to standard 0 for each neuron
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        # save inputs for backward pass
        self.inputs = inputs
        # multiply the inputs by the weights and add the biases for each batch
        outputs = inputs @ self.weights + self.biases
        return outputs

    def backward(self, dvalues):
        # calc the  partial derivative of each value
        self.dweights = np.dot(np.array(self.inputs).T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        del self.inputs
        return self.dinputs


# a convultional layer
class ConvolutionLayer:
    def __init__(self, kernels, height, width, depth, stride = 1, p = 0, innitlized = he):
        self.height = height
        self.width = width
        self.depth = depth
        self.weights = innitlized((kernels, height, width)) if depth == 1 else innitlized((kernels, depth, height, width))
        self.biases = np.zeros(kernels).astype(np.float32)
        self.stride = stride
        self.p = p

    def forward(self, inputs):

        self.inputs = np.array(inputs)

        if self.depth == 1:
            outputs = np.array([np.stack([scipy.signal.convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
                                 + bias for bias, kernel in zip(self.biases, self.weights)], axis=0)
                       for image in inputs])

            return outputs
        outputs = []

        for image in inputs:
            stack = []

            for bias, kernel in zip(self.biases, self.weights):
                conv_result = np.sum(
                    [scipy.signal.convolve2d(image[n], kernel[n], mode='same', boundary='fill', fillvalue=0) + bias
                     for n in range(kernel.shape[0])], axis=0)
                stack.append(conv_result)


            outputs.append(np.stack(stack, axis=0))


        outputs = np.array(outputs)
        return outputs

    def backward(self, dvalues):

        self.dweights = np.zeros_like(self.weights)
        if self.depth == 1:
            for image, dvalue in zip(self.inputs, dvalues):
                for i in range(image.shape[0]):
                    self.dweights[i] = scipy.signal.convolve2d(image, dvalue[i], mode='valid')
        else:
            for image, dvalue in zip(self.inputs, dvalues):
                for j in range(dvalue.shape[0]):
                    for i in range(image.shape[0]):
                        self.dweights[j, i] += scipy.signal.convolve2d(image[i], dvalue[j], mode='valid')

        self.dbiases = np.sum(dvalues, axis=(0, 2, 3))

        if self.depth == 1:
            flipped_weights = np.rot90(self.weights, 2)
            self.dinputs = [np.stack([scipy.signal.convolve2d(dvalue[j], flipped_weights[j], mode='same', boundary='fill', fillvalue=0)
                         for j in range(dvalue.shape[0])], axis=0)]
            del self.inputs
            return self.dinputs

        flipped_weights = np.rot90(self.weights, 2, axes=(2, 3))
        self.dinputs = []

        for image in dvalues:
            stack =[]
            for i in range(flipped_weights.shape[1]):
                stack.append(np.sum([scipy.signal.convolve2d(image[j], flipped_weights[j, i], mode='same', boundary='fill', fillvalue=0)
                            for j in range(image.shape[0])], axis=0))
            self.dinputs.append(np.stack(stack, axis=0))


        del self.inputs
        self.dinputs = np.array(self.dinputs)
        return self.dinputs


#max pooling layer
class PoolingLayer:
    def __init__(self, height, width, stride):
        self.dims_height, self.dims_width = height, width
        self.stride = stride

    def forward(self, inputs):
        self.height, self.width = inputs[0][0].shape

        height_out = (self.height - self.dims_height) // self.stride + 1
        width_out = (self.width - self.dims_height) // self.stride + 1

        final_max = []
        self.final_idx = []

        for image in inputs:

            stack_max = []
            stack_idx = []

            for channel in image:

                new_image = np.zeros((height_out, width_out))
                max_idx = []

                for h in range(height_out):
                    for w in range(width_out):

                        window = channel[h * self.stride:h * self.stride + self.dims_height, w * self.stride:w * self.stride + self.dims_width]
                        new_image[h][w] = np.max(window)
                        max_idx.append(np.argmax(window))

                stack_max.append(new_image)
                stack_idx.append(max_idx)

            final_max.append(np.stack(stack_max, axis=0))
            self.final_idx.append(np.stack(stack_idx, axis=0))
        return np.array(final_max)

    def backward(self, dvalues):
        dinputs = []
        for idx, image in zip(self.final_idx, dvalues):
            images = []
            for channel, idx in zip(image, idx):
                channel = channel.flatten()
                zeros = np.zeros(self.width * self.height)
                for i, v in zip(idx, channel):
                    zeros[i] = v
                zeros = zeros.reshape(self.width, self.height)
                images.append(zeros)
            dinputs.append(np.stack(images, axis=0))
        return dinputs


# global Average pooling Unfininshed
class GAP:
    def __init__(self):
        self.dims = None

    def forward(self, inputs):
        self.dims = inputs.shape
        return [[np.average(image[n]) for n in range(len(image.shape[4]))] for image in inputs]


# Dropout Layer FINISH
class DropoutLayer:
    def __init__(self, rate=0.3):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.mask = np.random.random(self.inputs.shape) > self.rate
        return inputs * self.mask

    def backward(self, dvalues):
        dvalues = dvalues * self.mask
        del self.mask, self.inputs
        return dvalues


#normilizes inputs for stability
class BatchNormalization:
    def __init__(self, momentum=0.1, epsilon=1e-8):
        self.m = momentum
        self.epsilon = epsilon
        self.mean = None
        self.var = None

    def forward(self, inputs):
        #check if it is the first pass
        if self.mean is None:
            #calc current mean & var
            self.mean = np.mean(inputs, axis=0)
            self.var = np.var(inputs, axis=0)
        else:
            # update the mean & var
            self.mean = self.m * self.mean + (1 - self.m) * np.mean(inputs, axis=0)
            self.var = self.m * self.var + (1 - self.m) * np.var(inputs, axis=0)
        return (inputs - self.mean) / np.sqrt(self.var + self.epsilon)

    def backward(self, dinputs):
        return dinputs * (1 / np.sqrt(self.var + self.epsilon))


# Sets any negative value to 0
class ReLU:
    def forward(self, inputs):
        # save inputs for backward pass
        self.inputs = inputs
        # set all negative numbers to zero
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        dinputs = dvalues * (np.array(self.inputs) > 0)
        del self.inputs
        return dinputs


# allows negative numbers, divided by 10 or * 0.1
class LeakyReLU:
    # forward pass, setting negative numbers to /10
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0.1 * inputs, inputs)

    # backward pass for LeakyReLU
    def backward(self, dvalues):
        dinputs = dvalues * (np.array(self.inputs) > 0) + 0.1 * dvalues * (np.array(self.inputs) < 0)
        del self.inputs
        return dinputs


# Exponential Linear Unit
class ELU:
    # set alpha value
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    # forward method
    def forward(self, inputs):
        # save inputs for backward pass
        self.inputs = inputs
        # return ELU function
        return np.where(inputs >= 0, inputs, self.alpha * np.exp(inputs) - 1)

    # backward pass
    def backward(self, dvalues):
        #calc the derivative
        deriv = np.where(self.inputs >= 0, 1, self.alpha * np.exp(self.inputs))
        # free self.inputs to save memory
        del self.inputs
        #return grads
        return deriv * dvalues


# scales the values between 1 & 0
class Sigmoid:
    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, dvalues):
        dinputs = self.outputs * (1 - self.outputs) * dvalues
        del self.outputs
        return dinputs


# Scales the values between -1 & 1
class Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        return np.tanh(inputs)

    def backward(self, dvalues):
        dvalues = dvalues * (1 - np.tanh(self.inputs) ** 2)
        del self.inputs
        return dvalues


# activation function for output layer
class Softmax:
    def forward(self, inputs):
        # Normilize inputs to prevent overflowing by subtracting maxium in the batch
        norm_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_inputs = np.exp(norm_inputs)
        # divide eulur's number to the i by the sum of all i values in the batch
        self.outputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.outputs

    def backward(self, dvalues):
        dinputs = np.empty_like(self.outputs)
        for i, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            softmax_jacobian = np.diag(single_output) - np.outer(single_output, single_output)
            dinputs[i] = np.dot(softmax_jacobian, single_dvalues)
        del self.outputs
        return dinputs


# Categorical Cross-entropy with One Hot encoded data
class CCE:
     def forward(self, model, predicated, expected, epsilon=1e-8):
         # clip values to prevent log of 0 errors in each batch
         clipped = np.clip(predicated, epsilon, 1 - epsilon)
         loss = -np.sum(expected * np.log(clipped)) / expected.shape[0]
         return loss

     def backward(self, predicated, expected, epsilon=1e-8):
         clipped = np.clip(predicated, epsilon, 1 - epsilon)
         dinputs = (clipped - expected) / expected.shape[0]
         return dinputs


# Mean Squared Error
class MSE:
    def forward(self, model, predicted, expected):
        loss = np.sum(np.square(predicted - expected)) / len(expected)
        return loss

    def backward(self, predicted, expected):
        return (2 / len(expected)) * (predicted - expected)


# optimizer
class Adam:
    def __init__(self, b1=0.9, b2=0.999):
        self.t = 0
        self.b1, self.b2 = b1, b2
        self.cached = {}

    def update(self, Layer, lr, epsilon=1e-8):
        # Initialize cache if not already done
        if id(Layer) not in self.cached:
            self.cached[id(Layer)] = {
                "mw": np.zeros_like(Layer.weights),
                "vw": np.zeros_like(Layer.weights),
                "mb": np.zeros_like(Layer.biases),
                "vb": np.zeros_like(Layer.biases)
           }

        # Increment step count
        self.t += 1

        # Retrieve cached momentums and velocities
        mw = self.cached[id(Layer)]["mw"]
        vw = self.cached[id(Layer)]["vw"]
        mb = self.cached[id(Layer)]["mb"]
        vb = self.cached[id(Layer)]["vb"]


        # Update momentums and velocities for weights
        mw = self.b1 * mw + (1 - self.b1) * Layer.dweights
        vw = self.b2 * vw + (1 - self.b2) * (Layer.dweights ** 2)

        # Update momentums and velocities for biases
        mb = self.b1 * mb + (1 - self.b1) * Layer.dbiases
        vb = self.b2 * vb + (1 - self.b2) * (Layer.dbiases ** 2)


        # Corrected momentums and velocities for weights
        mw_hat = mw / (1 - self.b1 ** self.t)
        vw_hat = vw / (1 - self.b2 ** self.t)
        mb_hat = mb / (1 - self.b1 ** self.t)
        vb_hat = vb / (1 - self.b2 ** self.t)

        # Update weights and biases
        Layer.weights = Layer.weights - lr * mw_hat / (np.sqrt(vw_hat) + epsilon)
        Layer.biases = Layer.biases - lr * mb_hat / (np.sqrt(vb_hat) + epsilon)

        # Save the updated momentums and velocities in the cache
        self.cached[id(Layer)] = {"mw": mw, "vw": vw, "mb": mb, "vb": vb}


# stops the model early when necessary
class EarlyStopping:
    def __init__(self, patience):
        self.patience, self.patience_counter = patience, 0
        self.best_model = None
        self.best_loss = float('inf')

    def __call__(self, model, dataset):
        new_loss = model.evaluate(dataset)
        if new_loss < self.best_loss:
            self.best_loss = new_loss
            self.best_model = model
        else:
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                self.stop(model)

    def stop(self, model):
       raise Exception("early stopping activated")
       model.stop = True


# increase then slowly decrease the lr for stability
class OneCycleLR:
    def __init__(self, max_lr, initial_lr, epochs, batch_size, dataset_size):
        self.total_steps = dataset_size / batch_size * epochs
        self.warmup_steps = self.total_steps * 0.1
        self.anneal_steps = self.total_steps - self.warmup_steps
        self.max_lr, self.initial_lr = max_lr, initial_lr
        self.lr = self.initial_lr

    def step(self, step):
        if step <= self.warmup_steps:
            self.warm_up(step)
        else:
            self.annealing(step)
        return self.lr

    def warm_up(self, step):
        self.lr = self.initial_lr + (step / self.warmup_steps) * (self.max_lr - self.initial_lr)

    def annealing(self, step):
        # Calculate the annealing rate (between 0 and 1)
        anneal = (step - self.warmup_steps) / self.anneal_steps
        # Update learning rate according to the annealing schedule
        self.lr = self.max_lr - (self.max_lr - self.initial_lr) * anneal


# slowely decay the lr lineary
class InverseTimeDecay:
    def __init__(self, lr=0.001, decay=5e-6):
        self.lr, self.initial_lr = lr, lr
        self.decay = decay

    def step(self, steps):
        if self.decay:
            self.lr = self.initial_lr / (1 + self.decay * steps)
        return self.lr


# layer that flattens tensors
class FlattenLayer:
    def forward(self, inputs):
        # stores the original shape for backward pass
        self.original_shape = np.array(inputs).shape
        # returns the reshaped matrix
        flattened_inputs = np.array(inputs).flatten()
        return np.split(flattened_inputs, self.original_shape[0])

    def backward(self, dinputs):
        dinputs = [np.split(image, self.original_shape[1]) for image in
                   np.split(dinputs.flatten(), self.original_shape[0])]
        del self.original_shape
        return dinputs


# overall model object that holds all data of the MLP
class Model:
    def __init__(self, layers: list = [], optimizer: object = Adam(), scheduler: object = InverseTimeDecay(0.01),
                 early_stopping: bool =None, loss: object =CCE(), dataset: object = None):
        # initialize each layer and activation function for forward and backward passes
        self.steps = layers
        # set the optimizer to Adam and pass the learning and decay rates
        self.optimizer = optimizer
        # set the loss function to Categorical Cross Entropy or Log Loss
        self.loss_function = loss
        # set whether early stopping is applicable
        self.early_stopping = early_stopping
        self.stop = False
        # set the scheduler
        self.scheduler = scheduler
        # set dataset
        self.dataset = dataset

    def add_layer(self, new_layer : object):
       self.layers.append(new_layer)

    def change_optimizer(self, new_optimizer : object):
        self.optimizer = new_optimizer

    def change_dataset(self, new_dataset : object):
        self.dataset = new_dataset

    def train(self, epochs : int, batch_size : int):
        for epoch in range(epochs):
            self.dataset.shuffle()
            # Calculate how many iterations to go through in one epoch
            for i in range(math.ceil(self.dataset.size[0] / batch_size)):
                # Create a batch of data and its corresponding truth values
                x_train, y_train = self.dataset.get_batch(batch_size, i)

                # Forward pass: Run through the network
                outputs = self._pass(self.steps, x_train)
                # Calculate loss gradients
                dinputs = self.loss_function.backward(outputs, y_train)

                # Backward pass of the network
                self._pass(self.steps[::-1],  dinputs, pass_type='backward')

                # Update weights and biases using the optimizer
                self._update()
                print(f'Epoch: {epoch + 1} {self.loss_function.forward(self, outputs, y_train)} Accuracy: '
                    f'{self._accuracy(outputs, y_train)} lr: {self.scheduler.lr}')

            #check if early stopping is enabled
            if self.early_stopping:
                self.early_stopping(self, self.dataset)
                if self.stop:
                    break

    def _update(self):
        # iterate through the steps, of the class is a dense layer then update the weights * biases using Adam
        for i in self.steps:
            if isinstance(i, DenseLayer) or isinstance(i, ConvolutionLayer):
                self.optimizer.update(i, self.scheduler.step(self.optimizer.t))

    def validate(self):
        # get the validation dataset and set it's truth values
        x, y_true = self.dataset.validate()
        # forward propagation inorder to determine what the ANN thinks
        self._pass(self.steps, x, False)
        return self.loss_function.forward(self, y_true)

    def test(self):
        # get the testing dataset and set it's truth values
        x, y_true = self.dataset.test()
        # forward propagation inorder to determine what the ANN thinks
        inputs = self._pass(self.steps,  x, False)
        # print out results
        print(f'Testing: {self.loss_function.forward(self, y_true, inputs)} lr: {self.scheduler.lr} '
                f'steps {self.optimizer.t} Accuracy: {self._accuracy(inputs, y_true)}')

    def _pass(self, steps: list, x_batch: np.array, training: bool = False, pass_type: str = 'forward'):
        # set the X_batch to inputs inorder to loop through each layer
        inputs = x_batch
        # call each step's forward method and set it's output to inputs
        for step in steps:
            # turns off Dropout Layers when on testing set or validation set
            if training or not isinstance(step, DropoutLayer):
                inputs = getattr(step, pass_type)(inputs)
        return inputs

    @staticmethod
    def _accuracy(y_pred, y_true):
        y_pred_indices = np.argmax(y_pred, axis=1)
        y_true_indices = np.argmax(y_true, axis=1)

        return np.mean(y_pred_indices == y_true_indices)

    def load_model(self, filename):
        # load and set the contents of a file to the current model
        with open(filename, 'rb') as file:
            self = pkl.load(file)

    def save_model(self, filename:str = 'model.pkl'):
        # save the model to a certain file
        with open(filename, 'wb') as file:
            pkl.dump(self, file)
