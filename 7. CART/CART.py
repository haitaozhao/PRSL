import pandas as pd
import numpy as np

# This program is modified from Josh Gordon's code on youtube

## Construction of Decision Tree including 4 things:
# 1. A set of questions (The answer is True or False)
# 2. The criterion for the partition of the tree and how to partition
# 3. When to stop
# 4. Attach classification information to the leaves of the tree

# 1. Question

class Question:
    """
    A Question is used to partition the training data.
    """
    def __init__(self,col,val):
    # Which column and what threshold to this column for partition
        self.col= col
        self.val = val

    def answer(self,data):
        # compare the feature value of certain column to the feature
        # value in this question
        if isinstance(self.val,int) or isinstance(self.val,float):
            # for int or float type
            result = data[self.col] >= self.val
        else:
            # for other type
            result = data[self.col] == self.val
        return result

    def __repr__(self):
        condition = '=='
        if isinstance(self.val,int) or isinstance(self.val,float):
            condition = '>='
        return 'Is {} {} {}?'.format(self.col,condition,str(self.val))

# 2. The criterion for the partition of the tree
def gini_impurity(y):
    # Computing the gini impurity only need consider the label information
	# impurity = 1 - \sum_{i=1}^C probability_i * probability_i
    temp = y.value_counts()
    impurity = 1
    for idx in temp.index:
        prob = temp[idx]/len(y)
        impurity -= prob**2
    return impurity

def info_gain(l_data, r_data, current_impurity):
    """Information Gain.
    The impurity of the starting node, minus the weighted impurity of two child nodes.
    input: l_data, r_data  are the labels of the left tree and the right tree respectively
    """
    p = float(len(l_data)) / (len(l_data) + len(r_data))
    return current_impurity - p * gini_impurity(l_data['label']) - (1 - p) * gini_impurity(r_data['label'])

## 2. (continued) How to partition the tree
# find the best gain and question for the partition
def find_best_split(data):
    """
    Find the best question to ask by iterating over every feature / value
    and calculating the information gain.
    """
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini_impurity(data['label'])
    n_features = list(data.columns)
    n_features.remove('label')

    for col in n_features:  # for each feature

        values = list(data[col].value_counts().index)

        for val in values:  # for each value

            q = Question(col, val)

            # try splitting the dataset
            true_data, false_data = data[q.answer(data)],data[~q.answer(data)]

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_data) == 0 or len(false_data) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_data, false_data, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, q

    return best_gain, best_question

class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """
    def __init__(self, data):
        self.predictions = data.label.value_counts()


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

## 3. Build the tree. Stopping partition till the gain is 0.
def build_tree(data):

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(data)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(data)

    # If we reach here, we have found a useful feature / value
    # to partition on.

    true_rows, false_rows = data[question.answer(data)],data[~question.answer(data)]

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        temp = node.predictions
        for idx in temp.index:
            #            print (spacing + "Predict", node.predictions)
            print(spacing + idx + ":  " + str(temp[idx]))
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

## 4. How to use the developed tree to test a new sample
def classify(example, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.answer(example):
        return classify(example, node.true_branch)
    else:
        return classify(example, node.false_branch)

def print_leaf(predictions):
    """A nicer way to print the predictions at a leaf."""
    total = predictions.sum()
    probs = {}
    for lbl in predictions.index:
        probs[lbl] = str(int(predictions[lbl] / total * 100)) + "%"
    return probs

if __name__== '__main__':
## The toy data for the demo of Decision Tree
# Each training data contains two features, color and sugar content
# The labels are three different juices, watermellon juice, cucumber juice and strawberry juice.
    data = [['yellow',6,'watermellon juice'],
        ['red',6,'watermellon juice'],
        ['green',2,'cucumber juice'],
        ['green',2,'cucumber juice'],
        ['red',6,'strawberry juice']]

    X_train = pd.DataFrame(data,columns=['color','sugar','label'])
# Build the tree and show the tree
    my_tree = build_tree(X_train)
    print_tree(my_tree)

## Testing

## The toy data for the demo of Decision Tree
# Each training data contains two features, color and sugar content
# The labels are three different juices, watermellon juice, cucumber juice and strawberry juice.
test = [['yellow',6,'watermellon juice'],
        ['red',8,'watermellon juice'],
        ['green',4,'cucumber juice'],
        ['green',2,'cucumber juice'],
        ['red',6,'strawberry juice']]
X_test = pd.DataFrame(test,columns=['color','sugar','label'])

for idx in range(len(X_test)):
    print('Sample Label: {}, Predicted Label: {}'.format(X_test.loc[idx,'label'],print_leaf(classify(X_test.loc[idx],my_tree))))