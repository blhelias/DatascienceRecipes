import os

import pandas as pd
from KMeans import KMeans


#################################### PATH VARIABLES #####################################
#========================================================================================
# DATA_PATH = "C:\\Users\\brieuc.lhelias\\Desktop\\workspace\\datascience_recipe\\data"
DATA_PATH = "C:\\Users\\brieu\\Desktop\\workspace\\machine learning from scratch\\10_algos_from_scratch\\data\\"
#========================================================================================
#########################################################################################

def load_kmeans_data():
    """
    charger le dataset data_1024.csv
    """
    data = pd.read_csv(os.path.join(DATA_PATH, 'data_1024.csv'), sep='\t')
    del data['Driver_ID']
    mean = data.mean(axis=0)
    data -= mean
    std = data.std(axis=0)
    data /= std
    return data.values

if __name__ == "__main__":
    X = load_kmeans_data()
#    instantiate KMeans class
    k_means = KMeans(n_clusters=4,
                     threshold=0.001,
                     n_iters = 1000,
                     initialization="forgy")
#    kmeans training
    k_means.fit(X)
    k_means.plot_training_history()