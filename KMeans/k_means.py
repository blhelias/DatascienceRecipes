"""
Created on Tue Sep 11 16:18:25 2018

@author: brieuc.lhelias
TODO: S'assurer qu'il n'y a pas de dupliqués lors de l'initialisation
de la forgy method.
TODO: Deplacer la methode load data
TODO: separer fit et plot
"""
import os
import math

from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import seaborn as sns

from path import DATA_PATH


class KMeans:
    """
    let's implement simple K-means !
    """
    def __init__(self, n_clusters: int, distance: str = "euclidian", 
                 threshold: float = 0.1, n_iters: int = 50, 
                 initialization: str = "forgy") -> None:

        self.n_clusters = n_clusters
        self.distance = distance
        self.threshold = threshold
        self.n_iters = n_iters
        self._initialization = initialization
        self._training_history = []
        self.fig = plt.figure("KMEANS")

    def initialization(self, X: np.ndarray, random: bool = False,
                       forgy: bool = False) -> np.array:
        """
        Cette fonction permet d'initialiser les centroids soit de maniere
        aléatoire ou en utilisant la technique de forgy
        """
        if self._initialization == "random":
            return np.random.rand(self.n_clusters, X.shape[1])
        elif self._initialization == "forgy":
            return [X[np.random.randint(0, X.shape[0])]
                    for i in range(self.n_clusters)]

    def voronoi_partition(self, liste_prototypes: np.array, 
            colors: List[str], X: np.ndarray) -> Tuple[Dict,List[str]]:
        """
        Cette methode prend en argument la liste des prototypes.
        A partir de cette liste, on va créer les nouveaux clusters,
        des sous partitions en fonction des distances de chaque point
        Avec les prototypes.
        """
        color_list = []
        partition: Dict = {}
#        Initialisation du dict
        for init_index in range(len(liste_prototypes)):
            partition[str(init_index)] = []

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
#            make sure the list is not empty before stacking rows
            if len(partition[str(min_dist_arg)]) != 0:
                partition[str(min_dist_arg)] = np.row_stack(
                    [partition[str(min_dist_arg)], data_point])
            else:
                partition[str(min_dist_arg)].append(data_point)
        return partition, color_list

    def compute_distance(self, pointa: np.array, pointb: np.array) -> float:
        """
        Distances entre 2 vecteurs
        """
        assert len(pointa) == len(pointb), "Arrays must be the same"\
                                           "dim ! "
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
                         previous_list: np.array) -> Tuple[np.array, float]:
        """
        renvoie le baricentre de chaque cluster
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
                    np.array(previous_list[element]))
        return np.array(centroids), dist

    def fit(self, X):
        """
        Training and vizualizing
        """
        colors = sns.color_palette(None, self.n_clusters)
        prototypes = self.initialization(X, forgy=True)
        for _ in range(self.n_iters):
            partition, color_list = self.voronoi_partition(prototypes, colors, X)
            prototypes, previous_dist = self.update_centroids(partition,
                                                              prototypes)
            print("centroids: {}".format(prototypes))
            a = plt.scatter(X[:, 0], X[:, 1], color=color_list, alpha=0.5)
            b = plt.scatter(prototypes[:, 0], prototypes[:, 1], color=colors, 
                                marker=">", edgecolor='black', s=100)
            self._training_history.append([a, b])
#            When the improvement of the algorith is bellow the threshold break 
            if previous_dist <= self.threshold:
                break
        return self
    
    def plot_training_history(self):
        img = animation.ArtistAnimation(self.fig, self._training_history, interval=500,
                                        blit=True, repeat_delay=100)
        plt.show()
        




if __name__ == "__main__":
    X = load_data()
#    instantiate KMeans class
    k_means = KMeans(n_clusters=4,
                     threshold=0.001,
                     n_iters = 1000,
                     initialization="forgy")
#    kmeans training
    k_means.fit(X)
    k_means.plot_training_history()
