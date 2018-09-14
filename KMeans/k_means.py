# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:18:25 2018

@author: brieuc.lhelias
TODO: ne pas reformater le dict a chaque itération !!!!
TODO: S'assurer quil n'y a pas de dupliqués lors de l'initialisation
de la forgy method.
"""
import math

from typing import Tuple, Dict

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


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
        centrer et réduire les donnees: souvent effectuer ce genre de technic
        diminue le temps de calcule de l'algo, ainsi que le résultat.
        """
        raise NotImplementedError

    def initialization(self,
                       random: bool = False,
                       forgy: bool = False) -> np.array:
        """
        # Cette fonction permet d'initialiser les centroids soit de maniere
        # aléatoire ou en utilisant la technique de forgy
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
        # Cette methode prend en argument la liste des prototypes.
        # A partir de cette liste, on va créer les nouveaux clusters,
        # des sous partitions en fonction des distances de chaque point
        # Avec les prototypes.
        """
        n = len(liste_prototypes)
        partition = {}
        for init_list in range(n):
            partition[str(init_list)] = []

        for data_point in self.data:
            min_dist: float = 100.
            min_dist_arg: int = 0
            for element in range(len(liste_prototypes)):
                temp_dist = self.compute_distance(data_point,
                                                  liste_prototypes[element])
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    min_dist_arg = element

            partition[str(min_dist_arg)].append(data_point)

        return self.reformat_dict_values(partition)

    def compute_distance(self,
                         pointa: np.array,
                         pointb: np.array) -> float:
        """
        # plusieurs distances sont en cours d'implementation
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

    def reformat_dict_values(self, dic: Dict) -> Dict:
        """
        le but est de supprimer des méthode qui rajoute
        des calcules en plus.
        """
        for key, value in dic.items():
            if dic[key] == []:
                continue
            else:
                dic[key] = np.vstack(value)
        return dic

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
        for _ in range(self.n_iters):
            partition = self.voronoi_partition(l_proto)
            l_proto, previous_dist = self.update_centroids(partition, l_proto)
            print("centroids: {}".format(l_proto))
            if l_proto.shape[1] == 2:
                plt.plot(self.data[:, 0], self.data[:, 1], 'o', color='b')
                plt.plot(l_proto[:, 0], l_proto[:, 1], 'X', color='r')
                plt.show()
            if previous_dist <= self.threshold:
                break
        return l_proto, partition

def generate_data(line: int, col: int):
    """
    générer un dataset random
    """
    return np.random.rand(line, col)

def load_data():
    """
    charger le dataset data_1024.csv
    """
    data = pd.read_csv('data/data_1024.csv', sep='\t')
    del data['Driver_ID']
    mean = data.mean(axis=0)
    data -= mean
    std = data.std(axis=0)
    data /= std
    return data.values

#cluster_data = generate_data(line=50, col=4)
cluster_data = load_data()  
#instantiate KMeans class
k_means = KMeans(n_clusters=4,
                 data=cluster_data,
                 distance="euclidian",
                 threshold=0.001,
                 n_iters = 100,
                 initialisation="forgy")
#kmeans training
centroids, partition_data = k_means.train()
