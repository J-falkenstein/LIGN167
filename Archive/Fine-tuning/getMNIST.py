from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models, layers, callbacks
import time
import numpy as np
import json

(train_X, train_y), (test_X, test_y) = mnist.load_data()


threshold = 125
def shapex(X):
    XX = np.empty_like(X)
    XX[X< threshold] = 0
    XX[X>=threshold] = 1
    return XX
train_X = shapex(train_X)
test_X = shapex(test_X)

dataset_size = 500
    
with open("dataset.txt", "w") as file:
    for sample_index in range(dataset_size):
        for row in train_X[sample_index]:
            for digit in row:
                file.write(str(digit))
            file.write("\n")
        file.write("\n")

data_dictionary = []
for sample_index in range(dataset_size): 
    number = train_y[sample_index]
    completion_matrix = ""
    for row in train_X[sample_index]:
        for digit in row:
            completion_matrix += str(digit)
        completion_matrix += "\n"
    completion_matrix = completion_matrix[:-1]
    dictonary_entry = dict(prompt = "Create a pixel matrix for a black and white image of the number " + str(number) +": \n\n###\n\n", completion =  " "+ completion_matrix +" END")
    data_dictionary.append(dictonary_entry)

with open("output.json", "w") as outfile:
    for entry in data_dictionary: 
        json.dump(entry, outfile)
        outfile.write('\n')