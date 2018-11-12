# Datascience Recipes

### Let's implement ML algorithms from scratch !!

## ACP

TODO: improve implementation of MyPCA class

## Linear regression

Few methods were tested such as OLS ans gradient descent.

```python
linear_regression = LinReg(method="gradient_descent")
linear_regression.fit(x, y)
print("linear_regression", linear_regression.coefs)
```

```
linear_regression [b, m] = [8.005218309921203, 1.3221519786061433] # intercept, slope
```

![alt text](https://i.imgur.com/LTrwkMk.png)

## Logistic Regression

### In this section, we build a logistic regression and compare our results with scikit learn.

```python
logreg = LogReg(lr=0.01)
logreg.fit(x, y) # estimate beta parameters
print(logreg.beta)
```

```
[-1.52102262  2.73661736 -2.64723693]
```

TODO :

  * Add some regularization
  * Try to implement logreg with scikit learn's cost function

## K-Means clustering

```python
k_means = KMeans(K_clusters=4, threshold=0.01, n_iters = 1000, initialization="forgy")
k_means.fit(X) #train model
k_means.plot_training_history(X)
```

![alt text](https://i.imgur.com/yeC5aJ4.gif)

## Decision Tree

Google tutorial with some few additionnals options

```python
tree = Tree()
my_tree = tree.fit(data) # build tree
tree.print_tree(my_tree)
```

Data                                                                                       |  Decision tree
:--------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:
![](https://github.com/blhelias/datascience_recipe/blob/master/doc/PlayTennis.jpg)  |  ![](https://github.com/blhelias/datascience_recipe/blob/master/doc/tree.png)


## SVM

TODO

## Gaussian Mixture Model

TODO
