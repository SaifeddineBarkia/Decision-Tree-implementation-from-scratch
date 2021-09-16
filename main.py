import pandas as pd
from decision_functions import *


# Initializing Parameters
alpha = 0.5
minNum = 5
default_class = 0
input_file = './data.csv'


# Reading data for use later 
data = pd.read_csv(input_file)


# Building the tree
Tree = BuildDecisionTree(input_file = input_file, minNum = minNum)
print("******** Built decision tree succefully ********\n")

# Printing the decision tree in the specified format
print("******** Decision Tree ********\n")
printDecisionTree(Tree, output_file = 'output_tree.txt')


# Calculating the generalization error for our data
print("******** Generalization error for our Tree ********\n")
generalizationError(data, Tree, alpha)


# Pruning the Tree and printing the result
pruneTree(Tree, alpha, minNum)
print("******** Pruned tree succefully, here's the Tree after pruning ********\n")
printDecisionTree(Tree, output_file = 'postpruned_tree.txt')