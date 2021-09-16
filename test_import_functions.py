import pandas as pd

alpha = 0.5
minNum = 5
data = pd.read_csv('./data.csv')
data = data[:20]

try:
	from decision_functions import BuildDecisionTree
	Tree = BuildDecisionTree(input_file = './data.csv', minNum = minNum)
	print('BuildDeciosionTree loaded!')
	print('----')
except Exception as e:
	raise e

try:
	from decision_functions import printDecisionTree
	printDecisionTree(Tree)
	print('printDecisionTree loaded!')
	print('----')
except Exception as e:
	raise e

try:
	from decision_functions import generalizationError
	generalizationError(data, Tree, alpha)
	print('generalizationError loaded!')
	print('----')
except Exception as e:
	raise e

try:
	from decision_functions import pruneTree
	pruneTree(Tree, alpha, minNum)
	print('pruneTree loaded!')
	print('----')
except Exception as e:
	raise e