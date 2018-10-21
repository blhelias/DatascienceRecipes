# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:18:25 2018

@author: brieuc.lhelias
TODO: ne pas reformater le dict a chaque itération !!!!
TODO: S'assurer quil n'y a pas de dupliqués lors de l'initialisation
de la forgy method.
"""
import os 

import math

from typing import Tuple, Dict

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
from path import DATA_PATH
import matplotlib.animation as animation

class KMeans:
    """
    let's implement simple K-means !
    """
    def __init__(self,
                 n_clusters: int,
                 data: np.ndarray,
                 distance: str = "euclidian",
                 threshold: float = 0.1,
                 n_iters: int = 50,
                 initialisation: str = "forgy") -> None:

        self.n_clusters = n_clusters
        self.data = data
        self.distance = distance
        self.threshold = threshold
        self.n_iters = n_iters
        self.initialisation = initialisation

    def normalization(self):
        """
        Centrer et réduire les données
        """
        raise NotImplementedError

    def initialization(self,
                       random: bool = False,
                       forgy: bool = False) -> np.array:
        """
        Cette fonction permet d'initialiser les centroids soit de maniere
        aléatoire ou en utilisant la technique de forgy
        """
        s = self.data.shape
        col = s[1]
        if self.initialisation == "random":
            return np.random.rand(self.n_clusters, col)
        elif self.initialisation == "forgy":
            liste = []
            for _ in range(self.n_clusters):
                liste.append(self.data[np.random.randint(0, self.data.shape[0])])
            return liste

    def voronoi_partition(self, liste_prototypes: np.array) -> Dict:
        """
        Cette methode prend en argument la liste des prototypes.
        A partir de cette liste, on va créer les nouveaux clusters,
        des sous partitions en fonction des distances de chaque point
        Avec les prototypes.
        """
        partition = {}
        for init_index in range(len(liste_prototypes)):
            partition[str(init_index)] = []

        for data_point in self.data:
            min_dist: float = 100.
            min_dist_arg: int = 0
            for element in range(len(liste_prototypes)):
                temp_dist = self.compute_distance(data_point,
                                                  liste_prototypes[element])
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    min_dist_arg = element
            # make sure the list is not empty before stacking rows
            if len(partition[str(min_dist_arg)]) != 0:
                partition[str(min_dist_arg)] = np.row_stack([partition[str(min_dist_arg)], 
                                                             data_point])
            else:
                partition[str(min_dist_arg)].append(data_point)

        return partition

    def compute_distance(self,
                         pointa: np.array,
                         pointb: np.array) -> float:
        """
        Distances entre 2 vecteurs
        """
        assert len(pointa) == len(pointb), "les deux arrays"\
        "n'ont pas la meme dim ! "

        distance = 0
        if self.distance == "euclidian":
            for element in range(len(pointa)):
                distance += (pointa[element] - pointb[element]) **2
            return math.sqrt(distance)
        elif self.distance == "manhattan":
            for element in range(len(pointa)):
                distance += abs(pointa[element] - pointb[element])
            return distance

    def update_centroids(self,
                         dic: Dict,
                         previous_list: np.array) -> Tuple[np.array, float]:
        """
        renvoie le baricentre de chaque cluster
        """
        dist: float = 0
        liste_update_centroids = []
        for _, value in dic.items():
            update_centroid = []
            if len(value) != 0:
                for i in range(value.shape[1]):
                    update_centroid.append(np.mean(value[:, i]))
                liste_update_centroids.append(update_centroid)
            for element in range(len(liste_update_centroids)):
                dist += self.compute_distance(\
                        np.array(liste_update_centroids[element]),
                        np.array(previous_list[element]))
        return np.array(liste_update_centroids), dist

    def train(self) -> Tuple[np.array, np.array]:
        """
        entrainement du modele
        """
        l_proto = self.initialization(forgy=True)
        partition = {}
        fig = plt.figure()
        ims = []
        for _ in range(self.n_iters):
            partition = self.voronoi_partition(l_proto)
            l_proto, previous_dist = self.update_centroids(partition, l_proto)
            print("centroids: {}".format(l_proto))
            if l_proto.shape[1] == 2:
                a, = plt.plot(self.data[:, 0], self.data[:, 1], 'o', color='b')
                b, = plt.plot(l_proto[:, 0], l_proto[:, 1], 'X', color='r')
                ims.append([a, b])
         
            if previous_dist <= self.threshold:
                break
        # Plot the training process with a nice animation
        img = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=100)
        plt.show()
        return l_proto, partition


def load_data():
    """
    charger le dataset data_1024.csv
    """
    data = pd.read_csv(os.path.join(DATA_PATH,'data_1024.csv'), sep='\t')
    del data['Driver_ID']
    mean = data.mean(axis=0)
    data -= mean
    std = data.std(axis=0)
    data /= std
    return data.values

if __name__ == "__main__":

    #cluster_data = generate_data(line=50, col=4)
    cluster_data = load_data()  
    #instantiate KMeans class
    k_means = KMeans(n_clusters=4,
                    data=cluster_data,
                    distance="euclidian",
                    threshold=0.001,
                    n_iters = 1000,
                    initialisation="forgy")
    #kmeans training
    centroids, partition_data = k_means.train()
