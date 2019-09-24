import rlcompleter, readline
readline.parse_and_bind('tab:complete')




from mlsquare.imly import dope
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_diabetes

model = LinearRegression()
diabetes = load_diabetes()

X = diabetes.data
sc = StandardScaler()
X = sc.fit_transform(X)
Y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)

m = dope(model)

m.fit(x_train, y_train)
m.score(x_test, y_test)

