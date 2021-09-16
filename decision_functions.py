import pandas as pd
from DecisionTree import DecisionTree

def BuildDecisionTree(input_file, minNum):

    data = pd.read_csv(input_file)
    
    Tree = DecisionTree(minNum = 5, d = 0)
    Tree.build(data = data, target = "Survived")
    
    return Tree

def printDecisionTree(Tree, output_file):

    output = ""    
    cur_level = 0
    
    for node, _ in Tree._bfs_traversal() :
        
        if node.level == cur_level and cur_level != 0:
            output += "*****"
        output += '\n'
        
        output += node.node_type + '\n'
        output += "Level " + str(node.level) + '\n'
        if node.node_type != "Leaf":
            output += "Feature " + str(node.feature) + ' ' + ' '.join([str(attr) for attr in Tree.attributes[node.feature] if attr < node.constr]) + '\n'
        else :
            output += "Class " + str(node.Class) + '\n'
        output += "Gini " + str(node.gini) + '\n'
        
        cur_level = node.level
    
    print(output)
    
    with open(output_file, 'w') as file :
        file.write(output)

def generalizationError(data, Tree, alpha):
    
    gen_error = Tree._generalization_error(data = data, alpha = alpha)
    
    with open('./generalization_error.txt', 'w') as file :
        file.write("Generilization error for the tree is {}".format(gen_error))
        
    return gen_error

def pruneTree(Tree, alpha, minNum):
	
    Tree._post_prune(alpha, minNum)
    
    return Tree
    
    
    
    
    
    
    