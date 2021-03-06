import data as data

class Question:
    """
    question used to partition data
    """
    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header
    
    def match(self, example):
        """
        compare the feature value with the example
        """
        val = example[self.column]
        if self.is_numeric(val):
            return val >= self.value
        else:
            return val == self.value
    
    def is_numeric(self, val):
        return isinstance(val, int) or isinstance(val, float)

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if self.is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))