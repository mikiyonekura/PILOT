# Load CSV
import numpy as np
import pandas as pd
import re

# load training and testing data from CSV files

def load_data(path, num_of_inputs, num_of_categories):
    """==============read training data=============="""
    raw_data = open(path+'/training_data.csv', 'rt')
    tr_d = np.loadtxt(raw_data, delimiter=",")
    training_inputs = [np.reshape(x, (num_of_inputs, 1)) for x in tr_d]
    raw_data = open(path+'/training_labels.csv', 'rt')
    tr_l = np.loadtxt(raw_data, delimiter=",")


    #train_labels_flat = train_data.iloc[:,0:1].values
    #train_labels_count = np.unique(tr_l).shape[0]


    training_labels = [vectorization(y,num_of_categories) for y in tr_l]
    training_data = zip(training_inputs, training_labels)


    """==============read testing data=============="""
    raw_data = open(path+'/testing_data.csv', 'rt')
    te_d = np.loadtxt(raw_data, delimiter=",")
    testing_inputs = [np.reshape(x, (num_of_inputs, 1)) for x in te_d]

    #test_labels = test_data.iloc[:,0:1].values
    #test_labels = dense_to_one_hot(test_labels, train_labels_count)
    #test_labels = test_labels.astype(np.uint8)


    test_data = pd.read_csv(path+'/testing_labels.csv',header = None)
    testing_labels = test_data.iloc[:,0:1].values
    testing_labels = dense_to_one_hot(testing_labels, num_of_categories)
    testing_labels = testing_labels.astype(np.uint8)


    #raw_data = open(path+'/testing_labels.csv', 'rt')
    #testing_labels = np.loadtxt(raw_data, delimiter=",")
    #testing_labels = dense_to_one_hot(testing_labels, num_of_categories)
    

    testing_data = testing_inputs
    #testing_data = zip(testing_inputs, te_l)

    return (training_data, testing_data, testing_labels)

def vectorization(j,num_of_categories):    
    e = np.zeros((num_of_categories, 1))
    e[int(j)] = 1.0
    return e

# Convert class labels from scalars to one-hot vectors 
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
