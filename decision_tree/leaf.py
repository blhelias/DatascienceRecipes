class Leaf:
    """a leaf node classifies data.

    this holds a dictionnary of class (e.g., "Apple) -> number of times
    it appears in the rows from the training data that reach this leaf
    """
    def __init__(self, rows):
        self.predictions = class_counts(rows)
    
def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row.target
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

    