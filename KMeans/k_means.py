"""
Created on Tue Sep 11 16:18:25 2018

@author: brieuc.lhelias
"""
import os
import math

from typing import Tuple, Dict, List

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


class KMeans:
    """K-means iplementation
    """

    def __init__(self, K_clusters: int, distance: str = "euclidian", 
                 threshold: float = 0.1, n_iters: int = 50, 
                 initialization: str = "forgy") -> None:

        self.K_clusters = K_clusters
        self.distance = distance
        self.threshold = threshold
        self.n_iters = n_iters
        self._initialization = initialization
        self._training_history = []
        

    def initialization(self, X: np.ndarray) -> np.array:
        """Initialize centroids randomly or using forgy method
        
        Args:
            X (np.ndarray): dataset       
        Returns:
            np.array: [description]
        """

        if self._initialization == "random":
            return np.random.rand(self.K_clusters, X.shape[1])
        elif self._initialization == "forgy":
            return [X[np.random.randint(0, X.shape[0])]
                    for i in range(self.K_clusters)]

    def voronoi_partition(self, liste_prototypes: np.array, 
            colors: List[str], X: np.ndarray) -> Tuple[Dict,List[str]]:
        """Cette methode prend en argument la liste des prototypes.
        A partir de cette liste, on va créer les nouveaux clusters,
        des sous partitions en fonction des distances de chaque point
        Avec les prototypes.
        """
        color_list = []
        partition: Dict = {}
        # Dict initialization with empty arrays
        for init_index in range(len(liste_prototypes)):
            partition[str(init_index)] = []
        # Give each point of the dataset a class
        for data_point in X:
            min_dist: float = 100.
            min_dist_arg: int = 0
            for element in range(len(liste_prototypes)):
                temp_dist = self.compute_distance(data_point,
                                                  liste_prototypes[element])
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    min_dist_arg = element
            color_list.append(colors[min_dist_arg])
            # Make sure the list is not empty before stacking rows
            if len(partition[str(min_dist_arg)]) != 0:
                partition[str(min_dist_arg)] = np.row_stack(
                    [partition[str(min_dist_arg)], data_point])
            else:
                partition[str(min_dist_arg)].append(data_point)
        return partition, color_list

    def compute_distance(self, pointa: np.array, pointb: np.array) -> float:
        """Compute the distance between 2 points
        
        Args:
            pointa (np.array)
            pointb (np.array)
        
        Raises:
            AttributeError: Make sure the distance is implemented
        
        Returns:
            float: distance
        """

        assert len(pointa) == len(pointb), "Arrays must be the same dim !"
        distance = 0
        if self.distance == "euclidian":
            for element in range(len(pointa)):
                distance += (pointa[element] - pointb[element]) ** 2
            return math.sqrt(distance)

        elif self.distance == "manhattan":
            for element in range(len(pointa)):
                distance += abs(pointa[element] - pointb[element])
            return distance

        raise AttributeError("The distance specified is invalid")

    def update_centroids(self, partition: Dict,
                         previous_centroids: np.array) -> Tuple[np.array, float]:
        """This method 
        
        Args:
            partition (Dict): voronoi partition
            previous_list (np.array): previous centroids
        
        Returns:
            Tuple[np.array, float]:
                - list_of_centroids
                - dist : distance from previous centroid
                  coor to update centroid coor. This variable 
                  is usefull to check how well our algo is converging.
        """

        dist: float = 0.
        centroids = []
        for _, value in partition.items():
            if len(value) != 0:
                update_centroid = [np.mean(value[:, i])
                                   for i in range(value.shape[1])]
                centroids.append(update_centroid)
            for element in range(len(centroids)):
                dist += self.compute_distance(
                    np.array(centroids[element]),
                    np.array(previous_centroids[element]))
        return np.array(centroids), dist

    def fit(self, X: np.ndarray):
        """train model
        
        Args:
            X (np.ndarray): dataset
        """

        colors = sns.color_palette(None, self.K_clusters)
        prototypes = self.initialization(X)
        for _ in range(self.n_iters):
            partition, color_list = self.voronoi_partition(prototypes, colors, X)
            prototypes, previous_dist = self.update_centroids(partition,
                                                              prototypes)
            print("centroids: {}".format(prototypes))
            # Ensure we can plot the data ( = 2 dimensions)
            if prototypes.shape[1] == 2:
                self._training_history.append([prototypes, color_list])
 #            Break when no major improvement 
            if previous_dist <= self.threshold:
                break
        return self
    
    def plot_training_history(self, X: np.ndarray):
        """Nice animation        
        Args:
            X (np.ndarray): dataset
        
        Raises:
            DimensionError: Make sure you use 2D dataset
        """

        colors = sns.color_palette(None, self.K_clusters)
        if self._training_history:
            fig = plt.figure("KMEANS")
            res = []
            for element in self._training_history:
                a = plt.scatter(X[:, 0], X[:, 1], color=element[1], alpha=0.5)
                b = plt.scatter(element[0][:, 0], element[0][:, 1], color=colors, 
                                marker=">", edgecolor='black', s=100)
                res.append([a, b])
            img = animation.ArtistAnimation(fig, res, interval=500,
                                            blit=True, repeat_delay=100)
            plt.show()
        else:
            raise DimensionError("Can not plot when dimension is greater than 2 !")


class DimensionError(Exception):
    pass