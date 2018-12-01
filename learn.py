import numpy as np
import csv
from math import sqrt

# Seed random number generator
np.random.seed(1)

# Define relu function
# Return x unless x lte 0
def relu(x):
    return (x > 0) * x

# Define function to return 
# slope / derivative of relu
# 1 unless relu 0
def relu_derivative(output):
    return output > 0

def build_matrix(file_name, column=-1):
  with open(file_name) as input_file:
    input_reader = csv.reader(input_file, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
    input_matrix = []

    for row in input_reader:
      if column == -1:
        input_matrix.append(row)
      else:
        row_val = row[column]
        input_matrix.append(row_val)

  return input_matrix

def merge_matrices(a, b, c):
  merged_matrix = []

  for iteration in range(len(a)):
    merged_matrix.append(a[iteration] + b[iteration] + c[iteration])
  return merged_matrix

# calculate column means
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means
 
# calculate column standard deviations
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs
 
# standardize dataset
def standardize_dataset(dataset, means, stdevs):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - means[i]) / stdevs[i]

# Setup learning rate
alpha = .01

# Setup architecture of hidden layer
hidden_size = 32

# Setup input data
hayward_data = 'data/hayward.csv'
livermore_data = 'data/livermore.csv'
sanjose_data = 'data/sanjose.csv'
fog_label = 'data/fog_label.csv'

# Build matricies for input data
hayward_matrix = build_matrix(hayward_data)
livermore_matrix = build_matrix(livermore_data)
sanjose_matrix = build_matrix(sanjose_data)

# Check that input sizes are the same
if(len(hayward_matrix) != len(livermore_matrix) or len(livermore_matrix) != len(sanjose_matrix)):
  raise Exception("Input matrix file size mismatch")

# Merge input data sources into single matrix
full_input = merge_matrices(hayward_matrix, livermore_matrix, sanjose_matrix)


means = column_means(full_input)
stdevs = column_stdevs(full_input, means)
standardize_dataset(full_input, means, stdevs)

full_input = np.array(full_input, dtype=float)

# Setup true values
labels = np.array(build_matrix(fog_label, column=1)).T

# Trim input to have same cut-off as input data
full_input = full_input[0:len(labels)]

# Track size of input matrix for network
# architecture
input_size = len(full_input[0])

# Check that label and input size are the same
if(len(labels) != len(full_input)):
  raise Exception("Input matrix and label value size mismatch")

# Setup initial weights
weights_0_1 = 2*np.random.random((input_size,hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size, 1)) - 1

# Setup learning iterator
# Pick 60 iterations
for iteration in range(1000):
  # Initialize container for 
  # total ouput error (layer 2)
  layer_2_error = 0
  # Iterate through input data rows
  for i in range(len(full_input)):
    # Initialize input layer with first 
    # row of input data
    layer_0 = full_input[i:i+1]
    # Generate layer 1 as dot product
    # of weights 0_1 and layer 0 and apply
    # relu function to activate/deactive
    layer_1 = relu(np.dot(layer_0, weights_0_1))
    # Generate layer 2 as dop product
    # of weights 1_2 and layer_1
    layer_2 = np.dot(layer_1, weights_1_2)
    # Calculate error of layer 2 output
    # By subtracting label true from prediction,
    # squaring, and adding to error variable
    layer_2_error += np.sum((labels[i:i+1] - layer_2) ** 2)
    # Calculate the delta between label true 
    # result and predicted
    layer_2_delta = labels[i:i+1] - layer_2
    # Calculate the error delta for layer 1
    # by multiplying layer 2 delta by layer 1 weights
    # and layer 1 relu derivative to see if perceptron activated
    layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu_derivative(layer_1)
    # Update weights by multiplying layer 1 
    # weights by error delta for layer 1_2
    # and moderating by learning rate alpha
    weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
    # Update weights by multiplaying layer 0
    # weights by error delata for layer 0_1
    # and moderating by learning rate alpha
    weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

  if(iteration % 10 == 9):
    print("Error[" + str(iteration) +"]: " + str(layer_2_error))