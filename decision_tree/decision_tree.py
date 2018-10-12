import itertools
from collections import namedtuple
from typing import List, NamedTuple

from split_criteria import Gini, Entropy
from data import data, Data
from question import Question
from leaf import Leaf, class_counts
from decision_node import Decision_Node

def get_total_proportion(data: NamedTuple, column: int, feat_name):
    count = 0
    for row in data:
        if row[column] == feat_name:
            count+=1
    return count / float(len(data))

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def partition(data, question):
    """Partition a dataset

    For each row in the dataset, 
    check if it matches the question or not
    """
    true_row, false_row = [], []
    for row in data:
        if question.match(row):
            true_row.append(row)
        else:
            false_row.append(row)
    return true_row, false_row

def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)
    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0.0:
        print("question = ", question)
        print(Leaf(rows).predictions)
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)
#    print("true_rows = ", true_rows)
#    print("false_rows = ", false_rows)
    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)

def find_best_split(data):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = Entropy().get_purity(data)
    n_features = len(data[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in data])  # unique values in the column
#        weighted_list = []
        for val in values:  # for each value
#            proportion = get_total_proportion(data, col, val)

            question = Question(col, val)
            # try splitting the dataset
            true_rows, false_rows = partition(data, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

#            weighted_impurity = Entropy().get_purity(true_rows) * proportion
#            weighted_list.append(weighted_impurity)
                
            #  Calculate the information gain from this split
            gain = Entropy().info_gain(true_rows, false_rows, current_uncertainty)
            # print(gain)
            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain > best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

def classify(row, node):
    """See the 'rules of recursion' above."""
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

if __name__ == "__main__":   
    my_tree = build_tree(data)
    for row in data:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_fruit_tree))))
    print_tree(my_tree)
    # for row in data:
    #     print ("Actual: %s. Predicted: %s" %
    #        (row[-1], print_leaf(classify(row, my_tree))))
    # print(classify(Data("sunny", "hot", "normal", "weak", "yes"), my_tree))
    
