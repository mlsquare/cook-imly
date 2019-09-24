import rlcompleter, readline
readline.parse_and_bind('tab:complete')

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

n = iris.data.shape[0]
for index in range(n):
	x = iris[index,]
	y = clf.predict(x)
