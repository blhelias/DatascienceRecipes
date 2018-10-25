import itertools
from collections import namedtuple
from typing import List, NamedTuple
#from split_criteria import Gini, Entropy
from data import data, Data
from question import Question
from leaf import Leaf, class_counts
from decision_node import Decision_Node


class Tree:
    def __init__(self, max_depth=10, min_samples_split=1, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def print_tree(self, node, spacing=""):
        """World's most elegant tree printing function."""

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print (spacing + "Predict", node.predictions)
            return

        # Print the question at this node
        print (spacing + str(node.question))

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")

    def partition(self, data, question):
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

    def build_tree(self, rows, depth=0):
        """Builds the tree.

        Rules of recursion: 1) Believe that it works. 2) Start by checking
        for the base case (no further information gain). 3) Prepare for
        giant stack traces.
        """
        # pre-pruning
        # max_depth, min_sample_split, min_bucket (min_samples_leaf)

        # Try partitioing the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        gain, question = self.find_best_split(rows)
        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if gain == 0.0:
            return Leaf(rows)

        # min_samples_split
        if len(rows) <= self.min_samples_split:
            return Leaf(rows)
        
        # max_depth
        if depth >= self.max_depth:
            return Leaf(rows)

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = self.partition(rows, question)

        # min_bucket/min_samples_leaf
        if len(true_rows) < self.min_samples_leaf or len(false_rows) < self.min_samples_leaf:
            return Leaf(rows)
        else:
            # Recursively build the true branch.
            true_branch = self.build_tree(true_rows, depth+1)

            # Recursively build the false branch.
            false_branch = self.build_tree(false_rows, depth+1)

            # Return a Question node.
            # This records the best feature / value to ask at this point,
            # as well as the branches to follow
            # depending on the answer.
        return Decision_Node(question, true_branch, false_branch, depth)

    def find_best_split(self, data):
        """Find the best question to ask by iterating over every feature / value
        and calculating the information gain."""
        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = Entropy().get_impurity(data)
        n_features = len(data[0]) - 1  # number of columns

        for col in range(n_features):  # for each feature

            values = set([row[col] for row in data])  # unique values in the column
            for val in values:  # for each value

                question = Question(col, val)
                # try splitting the dataset
                true_rows, false_rows = self.partition(data, question)

                # Skip this split if it doesn't divide the
                # dataset.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                    
                #  Calculate the information gain from this split
                gain = Entropy().info_gain(true_rows, false_rows, current_uncertainty)
                # print(gain)
                # You actually can use '>' instead of '>=' here
                # but I wanted the tree to look a certain way for our
                # toy dataset.
                if gain > best_gain:
                    best_gain, best_question = gain, question
        return best_gain, best_question

    def classify(self, row, node):
        """See the 'rules of recursion' above."""
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.predictions

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)

    def print_leaf(self, counts):
        """A nicer way to print the predictions at a leaf."""
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs

if __name__ == "__main__":
    tree = Tree() 
    my_tree = tree.build_tree(data)
    tree.print_tree(my_tree)

    
