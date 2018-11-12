class DecisionNode:
    """
    A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """
    def __init__(self,
                 question,
                 true_branch,
                 false_branch,
                 depth):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
    
    def __repr__(self):
        return "decision node question : %s, true branch : %s, false branch : %s, depth : %s>" % (self.question, 
                                                                                                  self.true_branch, 
                                                                                                  self.false_branch, 
                                                                                                  self.depth)
