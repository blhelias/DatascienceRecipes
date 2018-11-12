import pandas as pd
from ACP import PCA_scratch

def test_pca():
    # 1 Get the data in a dataframe
    dataset = pd.read_csv("data/Sales.csv", sep=";")
    del dataset["Unnamed: 0"]

    list(dataset.columns.values)

    # Clean the Sales column
    dataset.Sales = dataset.Sales.apply(lambda x: x.replace(',','.'))
    dataset = dataset.astype(float)

    #separate Sales target
    dataset.Sales = dataset.Sales.astype(float)
    target = dataset.Sales
    del dataset["Sales"]

    pca_homemade = PCA_scratch(n_components=5)
    U,S,V = pca_homemade.fit(dataset)
    transformed_data_handmade = pca_homemade.transform(dataset.values)
    print(transformed_data_handmade)
        
    # pca_sklearn = PCA(n_components=5)
    # pca_sklearn.fit(dataset.values)
    # #print(pca.explained_variance_ratio_)  
    # transformed_data=pca_sklearn.transform(dataset.values)
    # print(transformed_data)