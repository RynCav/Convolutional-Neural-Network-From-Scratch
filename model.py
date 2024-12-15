import math
import copy
from utils import *


# a single layer of a neural network
class DenseLayer:
    def __init__(self, n_inputs, n_neurons, l1=0, l2=0):
        # he initialized weights due to ReLU
        self.weights = [
            [random.gauss(0, math.sqrt(2.0 / n_inputs)) for i in range(n_neurons)] for i in range(n_inputs)]

        # set all biases in array to standard 0 for each neuron
        self.biases = [0 for i in range(n_neurons)]

        # set reg factors
        self.l1_factor, self.l2_factor = l1, l2

    def forward(self, inputs):
        # save inputs for backward pass
        self.inputs = inputs
        # multiply the inputs by the weights and add the biases for each batch
        outputs = [add_bias(dot_product(batch, self.weights), self.biases) for batch in inputs]
        return outputs

    def backward(self, dvalues):
        # calc the  partial derivative of each value
        self.dweights = matrix_multiply(transpose(self.inputs), dvalues)
        self.dbiases = [sum(d) for d in zip(*dvalues)]
        self.dinputs = matrix_multiply(dvalues, transpose(self.weights))


        # add the l1 gradients
        if self.l1_factor > 0:
            self.dweights = [[dw + self.l1_factor * (1 if w > 0 else -1) for dw, w in zip(drow, row)]
                             for drow, row in zip(self.dweights, self.weights)]
            self.dbiases = [db + self.l1_factor * (1 if b > 0 else -1) for db, b in zip(self.dbiases, self.biases)]

        # add the l2 gradients
        if self.l2_factor > 0:
            self.dweights = [[dw + 2 * self.l2_factor * w for dw, w in zip(drow, row)]
                             for drow, row in zip(self.dweights, self.weights)]
            self.dbiases = [db + 2 * self.l2_factor * b for db, b in zip(self.dbiases, self.biases)]

        return self.dinputs


# a convultional layer
class ConvolutionLayer:
    def __init__(self, kernels, dims, depth, stride, p):
        self.dims = dims
        self.depth = depth
        self.weights = [self._create_kernel() for i in range(kernels)]
        self.biases = [random.random() for i in range(kernels)]
        self.stride = stride
        self.p = p

    def forward(self, inputs):
        self.inputs = [self._padding(image) for image in inputs]

        return [self._add_bias(self._convolution(self._padding(image), kernel), bias)
                         for kernel, bias in zip(self.weights, self.biases)
                          for image in inputs]

    def backward(self, dinputs):
        kernels = [self._rotate_180(kernel) for kernel in self.weights]

        self.dweights = [[[self._calc_dweights(dinputs, [kernel.index(row), row.index(weight)])
                            for weight in row]
                            for row in kernel] for kernel in kernels]

        self.dbiases = [self._calc_dbiases(dinputs, kernel) for kernel in enumerate(kernels)]

        kernels = [[value for row in kernel for value in row] for kernel in kernels]

        return [self._convolution(self._padding(dimage), kernel)
                            for kernel in kernels
                            for dimage in dinputs]

    def _calc_dweights(self, dinputs, weight_pos):
        return sum(image[h][w] * dimage[h][w]
            for image, dimage in zip(self.inputs, dinputs)
            for h in range(weight_pos[0], min(len(image), weight_pos[0] + self.dims), self.dims)
            for w in range(weight_pos[1], min(len(image[0]), weight_pos[1] + self.dims), self.dims))

    def _calc_dbiases(self, dinputs, kernel_pos):
        return sum(dvalue
            for index, dimage in enumerate(dinputs)
            if index % len(self.weights) == kernel_pos
            for drow in dimage
            for dvalue in drow)

    def _padding(self, inputs):
        if self.p > 0:
            inputs = copy.deepcopy(inputs)

            if isinstance(inputs[0][0], list):
                channels = len(inputs[0])
                # adds 0's to the width
                [inputs[h].insert(0, zero_array(channels)) for h in range(len(inputs)) for i in range(self.p)]
                [inputs[h].append(zero_array(channels)) for h in range(len(inputs)) for i in range(self.p)]
                # adds 0's to the height
                inputs.insert(0, [zero_array(channels) for i in range(len(inputs[0]))])
                inputs.append([zero_array(channels) for i in range(len(inputs[0]))])

            else:
                # adds 0's to the width
                [inputs[h].insert(0, 0) for h in range(len(inputs)) for i in range(self.p)]
                [inputs[h].append(0) for h in range(len(inputs)) for i in range(self.p)]
                # adds 0's to the height
                [inputs.insert(0, [0] * len(inputs[0])) for i in range(self.p)]
                [inputs.append([0] * len(inputs[0])) for i in range(self.p)]
        # returns the padded image
        return inputs

    def _convolution(self, inputs, kernel):

        def multiply(list1, list2):
            if isinstance(list1[0], list):
                multiplied = [[c1 * c2 for c1, c2 in zip(l1, l2)] for l1, l2 in zip(list1, list2)]
                return [sum([item[channel] for item in multiplied]) for channel in range(len(multiplied[0]))]
            else:
                return sum([c1 * c2 for c1, c2 in zip(list1, list2)])

        # get the dims for the new image
        width = int(((len(inputs[0]) - self.dims) / self.stride + 1))
        height = int(((len(inputs) - self.dims) / self.stride + 1))
        return [[multiply([inputs[nh][nw]
                           for nh in range(h * self.stride, h * self.stride + self.dims)
                           for nw in range(w * self.stride, w * self.stride + self.dims)], kernel)
                 for w in range(width)]
                for h in range(height)]

    def _create_kernel(self):
        return [[random.uniform(-0.1, 0.1) for i in range(self.depth)] if self.depth > 1
                    else random.uniform(-0.1, 0.1) for i in range(self.dims ** 2)]

    def _rotate_180(self, inputs):
        inputs = split_list(inputs, self.dims)
        return [row[::-1] for row in inputs[::-1]]

    @staticmethod
    def _add_bias(inputs, bias):
        if isinstance([0][0], list):
            return [[[value + bias for value in channel] for channel in row] for row in inputs]
        else:
            return [[value + bias for value in row] for row in inputs]


# max pooling layer
class PoolingLayer:
    def __init__(self, dims, stride):
        self.dims = dims
        self.stride = stride

    def forward(self, inputs):
        self.inputs = inputs

        saved_outputs = []
        saved_indices = []
        for image in inputs:

            new_image, new_indices = self._pooling(self._padding(image))

            saved_outputs.append(new_image)
            saved_indices.append(new_indices)

        self.outputs = saved_outputs
        self.indices = saved_indices

        return self.outputs

    def backward(self, inputs):
        return [self._input_pos(indices, image) for indices, image in zip(self.indices, inputs)]

    def _padding(self, inputs):
        inputs = copy.deepcopy(inputs)
        while len(inputs[0]) / self.dims != int(len(inputs[0]) / self.dims):
            [inputs[i].append([0] * len(inputs[0][0])) if isinstance(inputs[0][0], list) else inputs[i].append(0)
            for i in range(len(inputs))]

        added_zeros = 0
        while (len(inputs[0]) + added_zeros) / self.dims != int((len(inputs[0]) + added_zeros) / self.dims):
            added_zeros += 1
            inputs.append([0] * len(inputs[0]) + [zero_array(len(inputs[0][0])) for i in range(added_zeros)]
                            if isinstance(inputs[0][0], list) else [0 for i in added_zeros])
        return inputs

    def _pooling(self, inputs):
        """complete this"""
        def channel_max(window):
            # Multi-channel: each channel has its own max
            channels = len(window[0][0])  # Assuming all windows are consistent
            max_values = [max(row[channel] for subwindow in window for row in subwindow) for channel in range(channels)]
            max_indices = [[(nh, nw, channel) for nh in range(len(window)) for nw in range(len(window[0]))
                 if window[nh][nw][channel] == max_values[channel]]
                for channel in range(channels)]
            return max_values, max_indices

        def single_max(window):
            max_value = max(value for row in window for value in row)
            for nh in range(len(window)):
                for nw in range(len(window[0])):
                    if window[nh][nw] == max_value:
                        max_index = [nh, nw]
                        break
            return max_value, max_index

        # Get the dimensions for the new image
        height = (len(inputs) - self.dims) // self.stride + 1
        width = (len(inputs[0]) - self.dims) // self.stride + 1

        pooled_image = []
        pooled_indices = []


        for h in range(height):
            pooled_row = []

            for w in range(width):
                # Extract window
                window = [inputs[nh][w * self.stride:w * self.stride + self.dims]
                          for nh in range(h * self.stride, h * self.stride + self.dims)]

                if isinstance(window[0][0], list):  # Multi-channel
                    max_values, max_indices = channel_max(window)

                    max_indices[0]  = h * self.stride + max_indices[0]
                    max_indices[1] = w * self.stride + max_indices[1]
                    max_indices[2] = max_indices[2]

                    pooled_indices.append(max_indices)
                    pooled_row.append(max_values)

                else:
                    max_value, max_indices = single_max(window)

                    max_indices[0] = (h * self.stride + max_indices[0])
                    max_indices[1] = (w * self.stride + max_indices[1])

                    pooled_indices.append(max_indices)
                    pooled_row.append(max_value)

            pooled_image.append(pooled_row)

        return pooled_image, pooled_indices

    def _input_pos(self, indices, inputs):

        def zero_image(inputs):
            return [[[0 for c in l] if isinstance(l, list) else 0 for l in i] for i in self.inputs[0]]
        dimage = zero_image(inputs)

        if isinstance(inputs[0][0], list):
            inputs = [max(l) for i in inputs for l in i]
            for value_indices, number in zip(indices, inputs):
                dimage[value_indices[0]][value_indices[1]][value_indices[2]] = number
        else:
            inputs = [l for i in inputs for l in i]
            for value_indices, value in zip(indices, inputs):
                dimage[value_indices[0]][value_indices[1]] = value

        return dimage


# Dropout Layer
class DropoutLayer:
    def __init__(self, rate=0.3):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.bmask = [[1 if random.random() > self.rate else 0 for i in row] for row in inputs]
        outputs = [[i * m for i, m in zip(irow, mrow)] for irow, mrow in zip(inputs, self.bmask)]
        return outputs

    def backward(self, dvalues):
        dinputs = [[d * m for d, m in zip(drow, mrow)] for drow, mrow in zip(dvalues, self.bmask)]
        return dinputs


# sets any negative value to 0
class ReLU:
    def forward(self, inputs):
        # save inputs for backward pass
        self.inputs = inputs
        # set all negative numbers to zero
        return [[[[max(0, value) for value in row]
                   if isinstance(row, list) else max(0, row)
                   for row in number]
                 if isinstance(number, list) else max(0, number) for number in image]
                 for image in inputs]

    def backward(self, dvalues):
        return [[[[dvalue if value > 0 else 0 for dvalue, value in zip(drow, row)]
                  if isinstance(row, list) else drow if row > 0 else 0
                  for drow, row in zip( dnumber, number)]
                 if isinstance(number, list) else dnumber if number > 0 else 0
                 for dnumber, number in zip(dimage, image)]
                for dimage, image in zip(dvalues, self.inputs)]

# scales the values between 1 & 0
class Sigmoid:
    def forward(self, inputs):
        self.outputs = [[[[1 / (1 + math.exp(-number) for number in channel)]
                            if isinstance(channel, list) else 1 / (1 + math.exp(-channel))
                           for channel in row] for row in value]
                         if isinstance(value, list) else 1 / (1 + math.exp(-value))
                         for value in inputs]
        return self.outputs

    def backward(self, dvalues):
        dinputs = [[[[value * (1 - value) * dvalue for value, dvalue in zip(channel, dchannel)]
                      if isinstance(channel, list) else channel * (1 - channel) * dchannel
                      for channel, dchannel in zip(row, drow)]
                    if isinstance(output, list) else output * (1 - output) * dvalue
                    for output, dvalue in zip(batch, dbatch)]
        for batch, dbatch in zip(self.outputs, dvalues)]

        return dinputs


# Scales the values between -1 & 1
class Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        outputs = [[[[[math.tanh(value) for value in channel]
                      if isinstance(channel, list) else math.tanh(channel)
                      for channel in row] for row in i]
                    if isinstance(i, list) else math.tanh(i)
                    for i in batch] for batch in inputs]
        return outputs

    def backward(self, dvalues):
        dinputs = [[[[[dvalue * (1 - math.tanh(value) ** 2) for value, dvalue in zip(channel, dchannel)]
                      if isinstance(channel, list)
                      else dchannel * (1 - math.tanh(channel) ** 2)
                     for channel, dchannel in zip(row, drow)]
                    for row, drow in zip(i, d)]
                    if isinstance(d, list) else d * (1 - math.tanh(i) ** 2)
                   for i, d in zip(batch, dbatch)] for batch, dbatch in
                   zip(self.inputs, dvalues)]
        return dinputs


# activation function for output layer
class Softmax:
    def forward(self, inputs):
        # Normilize inputs to prevent overflowing by subtracting maxium in the batch
        norm_inputs = [[batch[i] - max(batch) for i in range(len(batch))] for batch in inputs]
        # divide eulur's number to the i by the sum of all i values in the batch
        self.outputs = [[math.exp(i) / sum([math.exp(i) for i in batch]) for i in batch] for batch in norm_inputs]
        return self.outputs

    """impove this."""
    def backward(self, dvalues):
        # Initialize a list for gradients with the same structure as dvalues
        dinputs = [[0] * len(batch) for batch in dvalues]

        # Iterate over each sample in the batch
        for sample_index, (output, dvalue) in enumerate(zip(self.outputs, dvalues)):
            # For each sample, iterate over each output i (rows)
            for i in range(len(output)):
                # Initialize the gradient for this output to 0
                gradient_sum = 0

                # For each output j (columns), calculate the gradient
                for j in range(len(output)):
                    # Softmax derivative: For i == j, use output_i * (1 - output_i)
                    # For i != j, use -output_i * output_j
                    if i == j:
                        gradient = output[i] * (1 - output[j])
                    else:
                        gradient = -output[i] * output[j]

                    # Multiply by the incoming gradient (from the next layer)
                    gradient_sum += gradient * dvalue[j]

                # Store the computed gradient in the corresponding place
                dinputs[sample_index][i] = gradient_sum
        return dinputs


#L1 and L2 Reg Loss
class _RegLoss:
    def regulazation_loss(self, model):
        reg_loss = 0
        for step in model.steps:
            if isinstance(step, DenseLayer):
                if step.l1_factor > 0:
                    reg_loss += step.l1_factor * sum(sum(abs(w) for w in row) for row in step.weights)
                if step.l2_factor > 0:
                    reg_loss += step.l2_factor * sum(sum(w ** 2 for w in row) for row in step.weights)
        return reg_loss


# Categorical Cross-entropy with One Hot encoded data
class CCE(_RegLoss):
     def forward(self, model, predicated, expected, epsilon=1e-8):
         # clip values to prevent log of 0 errors in each batch
         clipped = [[max(epsilon, min(i, 1 - epsilon)) for i in batch] for batch in predicated]
         loss_matrix = [sum([-math.log(l) if e == 1 else 0 for l, e in zip(b, e)]) for b, e in
                       zip(clipped, expected)]
         # calc & return the losses and add reg loss
         norm_loss, reg_loss = sum(loss_matrix) / len(loss_matrix)
         return f'Loss: {norm_loss} (Normal Loss: {norm_loss} Reg Loss: {1})'

     def backward(self, predicated, expected, epsilon=1e-8):
         clipped = [[max(epsilon, min(i, 1 - epsilon)) for i in batch] for batch in predicated]
         dinputs = [[(c - e) for c, e in zip(cl, ex)] for cl, ex in zip(clipped, expected)]
         return dinputs


# Mean Squared Error
class MSE(_RegLoss):
    def forward(self, model, predicted, expected):
        norm_loss = sum([sum([(y - y_hat) ** 2 for y, y_hat in zip(batch, batch_hat)]) / len(batch)
                         for batch, batch_hat in zip(predicted, expected)]) / len(expected)
        reg_loss = self.regulazation_loss(model)
        return f'Loss: {norm_loss + reg_loss} (Normal Loss: {norm_loss} Reg Loss: {reg_loss})'

    def backward(self, predicted, expected):
        N = len(expected)
        dinputs = [[(2 / N) * (y_hat - y) for y_hat, y in zip(batch_hat, batch)]
                        for batch_hat, batch in zip(predicted, expected)]
        return dinputs


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
                "mw": [[[0 for p in i] if isinstance(Layer, ConvolutionLayer) else 0 for i in l] for l in Layer.dweights],
                "vw": [[[0 for p in i] if isinstance(Layer, ConvolutionLayer) else 0 for i in l] for l in Layer.dweights],
                "mb": [0] * len(Layer.biases),
                "vb": [0] * len(Layer.biases)
           }

        # Increment step count
        self.t += 1

        # Retrieve cached momentums and velocities
        mw = self.cached[id(Layer)]["mw"]
        vw = self.cached[id(Layer)]["vw"]
        mb = self.cached[id(Layer)]["mb"]
        vb = self.cached[id(Layer)]["vb"]


        # Update momentums and velocities
        mw = [[[self.b1 * m_v + (1 - self.b1) * dw_v for m_v, dw_v in zip(m, dw)] if isinstance(Layer, ConvolutionLayer)
               else self.b1 * m + (1 - self.b1) * dw for m, dw in zip(mw, dweights)]
              for mw, dweights in zip(mw, Layer.dweights)]
        vw = [[[self.b2 * v_v + (1 - self.b2) * (dw_v ** 2) for dw_v, v_v in zip(dw, v)]
               if isinstance(Layer, ConvolutionLayer)
               else self.b2 * v + (1 - self.b2) * (dw ** 2)
               for dw, v in zip(dweights, vw)]
              for dweights, vw in zip(Layer.dweights, vw)]


        mb = [self.b1 * mb[i] + (1 - self.b1) * Layer.dbiases[i] for i in range(len(mb))]
        vb = [self.b2 * vb[i] + (1 - self.b2) * (Layer.dbiases[i] ** 2) for i in range(len(vb))]


        # Corrected momentums and velocities
        mw_hat = [[[m_value / (1 - self.b1 ** self.t) for m_value in m]
                  if isinstance(Layer, ConvolutionLayer)
                  else m / (1 - self.b1 ** self.t) for m in mw_row] for mw_row in mw]
        vw_hat = [[[v_value / (1 - self.b2 ** self.t) for v_value in v] if isinstance(Layer, ConvolutionLayer)
                  else v / (1 - self.b2 ** self.t) for v in vw_row] for vw_row in vw]
        mb_hat = [m / (1 - self.b1 ** self.t) for m in mb]
        vb_hat = [v / (1 - self.b2 ** self.t) for v in vb]


        # Update weights and biases
        if isinstance(Layer, ConvolutionLayer):
            weights = [split_list(row, len(mw_hat[0])) for row in Layer.weights]
        else:
            weights = Layer.weights
        Layer.weights = [[[w1 - lr * m1 / (math.sqrt(v1) + epsilon) for w1, m1, v1 in zip(w, m, v)]
                          if isinstance(Layer, ConvolutionLayer)
                  else w - lr * m / (math.sqrt(v) + epsilon) for w, m, v in zip(weights, mh, vh)]
                         for weights, mh, vh in zip(weights, mw_hat, vw_hat)]
        if isinstance(Layer, ConvolutionLayer):
            Layer.weights = [[value for row in kernel for value in row] for kernel in Layer.weights]
        Layer.biases = [b - lr * m / (math.sqrt(v) + epsilon) for b, m, v in
                        zip(Layer.biases, mb_hat, vb_hat)]

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
    def __init__(self, lr=0.01, decay=5e-5):
        self.lr, self.initial_lr = lr, lr
        self.decay = decay

    def step(self, steps):
        if self.decay:
            self.lr = self.initial_lr / (1 + self.decay * steps)
        return self.lr


# layer that flattens tensors
class FlattenLayer:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def forward(self, inputs):
        inputs = split_list(inputs, self.batch_size)
        self.rows, self.images = len(inputs[0][0][0]), len(inputs[0][0])
        return [[value for image in image_set for row in image for value in row] for image_set in inputs]

    def backward(self, dinputs):
        self.dinputs = [value for image in dinputs for value in image]
        for split in [self.rows, self.images]:
            self.dinputs = self._split_n(self.dinputs, split)
        return self.dinputs

    @staticmethod
    def _split_n(inputs, n):
        if n > 1:
            return [inputs[i:i + n] for i in range(0, len(inputs), n)]
        return inputs


# overall model object that holds all data of the MLP
class Model:
    def __init__(self, layers=[], optimizer=Adam(), scheduler=InverseTimeDecay(0.01), early_stopping=None, loss=CCE()):
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

    def add(self, new_layer):
       self.layers.append(new_layer)

    def update(self):
        # iterate through the steps, of the class is a dense layer then update the weights * biases using Adam
        for i in self.steps:
            if isinstance(i, DenseLayer) or isinstance(i, ConvolutionLayer):
                self.optimizer.update(i, self.scheduler.step(self.optimizer.t))

    def train(self, dataset, epochs, batch_size, auto_encoder=False):
        x, y = dataset.train()
        if auto_encoder:
            y = x
        for epoch in range(epochs):
            # Calculate how many iterations to go through in one epoch
            for i in range(math.ceil(len(x) / batch_size)):
                # Create a batch of data and its corresponding truth values
                x_train, y_train = batch(x, batch_size, i), batch(y, batch_size, i)
                # Forward pass: Run through the network

                outputs = self.n_pass(self.steps, 'forward', x_train, True)
                # Calculate loss gradients
                dinputs = self.loss_function.backward(outputs, y_train)

                # Backward pass of the network
                self.n_pass(self.steps[::-1], 'backward', dinputs, True)

                # Update weights and biases using the optimizer
                self.update()

            #print infomation
            print(f'Epoch: {epoch + 1} {self.loss_function.forward(self, outputs, y_train)} Accuracy: '
                        f'{self.accuracy(outputs, y_train)} lr: {self.scheduler.lr}')

            if self.early_stopping:
                self.early_stopping(self, dataset, auto_encoder)
                if self.stop:
                    break

    def validate(self, dataset, auto_encoder=False):
        # get the validation dataset and set it's truth values
        x, y_true = dataset.validate()
        if auto_encoder:
            y_true = x
        # forward propagation inorder to determine what the ANN thinks
        self.n_pass(self.steps, 'forward', x, False)
        return self.loss_function.forward(self, y_true)

    def test(self, dataset, auto_encoder=False):
        # get the testing dataset and set it's truth values
        x, y_true = dataset.test()
        if auto_encoder:
            y_true = x
        # forward propagation inorder to determine what the ANN thinks
        inputs = self.n_pass(self.steps, 'forward', x, False)
        # print out results
        print(f'Testing: {self.loss_function.forward(self, y_true, inputs)} lr: {self.scheduler.lr} '
                f'steps {self.optimizer.t} Accuracy: {self.accuracy(inputs, y_true)}')

    def n_pass(self, steps, pass_type, x_batch, training):
        # set the X_batch to inputs inorder to loop through each layer
        inputs = x_batch
        # call each step's forward method and set it's output to inputs
        for step in steps:
            # turns off Dropout Layers when on testing set or validation set
            if training or not isinstance(step, DropoutLayer):
                inputs = getattr(step, pass_type)(inputs)
        return inputs

    @staticmethod
    def accuracy(y_pred, y_true):
        predicted_labels = [argmax(p) for p in y_pred]
        true_labels = [argmax(t) for t in y_true]

        # Calculate the number of correct
        correct = sum([1 if pred == true else 0 for pred, true in zip(predicted_labels, true_labels)])

        # Return accuracy
        return correct / len(y_true)
