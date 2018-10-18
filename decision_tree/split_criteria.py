from leaf import class_counts
import math

class SplitCriteria:

    def get_impurity(self, data):
        raise NotImplementedError
    
    def info_gain(self, left, right, current_uncertainty):
        raise NotImplementedError


class Gini(SplitCriteria):

    def get_impurity(self, data):
        """
        sum (p_j * ( 1 - p_j )) = 1 - sum(pj)
        """
        sum_squared_p_i = 0
        dict_class = class_counts(data)
        data = [x.target for x in data]
        for class_i in dict_class:
            p_i = dict_class[class_i] / float(len(data))
            sum_squared_p_i += p_i ** 2
        res = 1 - sum_squared_p_i
        print(res)
        return res

    def info_gain(self, left, right, current_uncertainty):
        """information gain
        
        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        """
        p_l = float(len(left)) / (len(left) + len(right))
        p_r = 1 - p_l
        return current_uncertainty - p_l * self.get_impurity(left) - p_r * self.get_impurity(right)


class Entropy(SplitCriteria):

    def get_impurity(self, data):
        """
        ( - sum ( p_j * log( p_j ) ) )
        """
        sum_squared_p_i = 0
        dict_class = class_counts(data)
        data = [x.target for x in data]
        for class_i in dict_class:
            p_i = dict_class[class_i] / float(len(data))
            sum_squared_p_i -= p_i * math.log2(p_i)
        res = sum_squared_p_i
        return res

    def info_gain(self, left, right, current_uncertainty):
        """information gain
        
        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        """
        p_l = float(len(left)) / (len(left) + len(right))
        p_r = 1 - p_l
        return current_uncertainty - p_l * self.get_impurity(left) - p_r * self.get_impurity(right)


