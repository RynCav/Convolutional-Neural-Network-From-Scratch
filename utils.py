import pickle
import random


def dot_product(array, matrix):
    # returns the dot product of an array and a matrix passed
    return [sum(array[i] * matrix[i][n] for i in range(len(array))) for n in range(len(matrix[0]))]


def add_bias(matrix, biases):
    # adds the contents of two arrays indexes together
    return [a + b for a, b in zip(matrix, biases)]


def transpose(matrix):
    # return the opisite dimisions of a matrix/array
    return [[r[i] for r in matrix] for i in range(len(matrix[0]))]


def matrix_multiply(matrix1, matrix2):
    # get the number of rows and columns of the matrices

    num_rows_matrix1 = len(matrix1)
    num_cols_matrix2 = len(matrix2[0])
    num_cols_matrix1 = len(matrix1[0])
    #multiply the two matrices
    result = [[0] * num_cols_matrix2 for _ in range(num_rows_matrix1)]
    for i in range(num_rows_matrix1):
        for j in range(num_cols_matrix2):
            result[i][j] = sum(matrix1[i][k] * matrix2[k][j] for k in range(num_cols_matrix1))
    #return the new matrix
    return result


def batch(X, batch_size, i):
    # returns a batch of n size that is conatins data that hasnt been trained before in the Epoch
    return [X[l] for l in range(i * batch_size, i * batch_size + batch_size)]


def argmax(list):
    # return the highest value's index
    return list.index(max(list))


def randomize_lists(list1, list2):
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    return zip(*combined)


#produce a list of length n that contains zeros
def zero_array(channels):
   return [0 for i in range(channels)] if channels > 1 else 0

#splits a list into n sublists of equal length
def split_list(inputs, split):
    if split > 1:
        sublist_length = len(inputs) // split
        return [inputs[i * sublist_length:(i + 1) * sublist_length] for i in range(split)]
    return inputs


"""Load and save the model"""
def load(filename):
    # load and set the contents of a file equal to a varable / object
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save(savedable, filename):
    # save the model to a certain file
    with open(filename, 'wb') as file:
        pickle.dump(savedable, file)
