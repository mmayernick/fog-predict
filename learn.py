import numpy as np
import csv

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
    input_reader = csv.reader(input_file, delimiter=',')
    input_matrix = []

    for row in input_reader:
      if column == -1:
        input_matrix.append(row)
      else:
        # Convert T/F to int
        if row[column] == "True":
          row_val = 1
        elif row[column] == "False":
          row_val = 0
        else:
          row_val = row[column]
        input_matrix.append(row_val)

  return input_matrix

def merge_matrices(a, b, c):
  merged_matrix = []

  for iteration in range(len(a)):
    # Skip header row
    if iteration > 0:
      merged_matrix.append(a[iteration] + b[iteration] + c[iteration])

  return merged_matrix

# Setup learning rate
alpha = 0.2

# Setup architecture of hidden layer
hidden_size = 20

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
full_input = np.array(merge_matrices(hayward_matrix, livermore_matrix, sanjose_matrix))

# Setup true values
labels = np.array(build_matrix(fog_label, column=0)).T

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
for iteration in range(60):
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
        print("Error: " + str(layer_2_error))