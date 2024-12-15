from utils import save, load, randomize_lists
import numpy

# inherited class, template
class Data:
    # initialize the data
    def __init__(self, directory, valuation_set=False):
        # load files, values are already normilized & y_true is onehot encoded
        self.train_X, self.test_X = load(f'{directory}x_train.pkl'), load(f'{directory}x_test.pkl')
        self.train_y, self.test_y = load(f'{directory}y_train.pkl'), load(f'{directory}y_test.pkl')
        # get the size of the dataset
        self.size = len(self.train_X)
        # load the validation set if applicable
        if valuation_set:
            self.evaluate_y, self.evaluate_y = load(f'{directory}Evaluate_y'), load(f'{directory}Evaluate_y')

    # load the training data
    def train(self):
        # Randomises the data
        return randomize_lists(self.train_X, self.train_y)

    def evaluate(self):
        return self.evaluate_X, self.evaluate_y

    # load the testing data
    def test(self):
        # returns the testing data
        return self.test_X, self.test_y


# Define the datasets
FashionMNIST = Data('Datasets/Fashion MNIST/')

