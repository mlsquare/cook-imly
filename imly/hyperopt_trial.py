# from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def data():
    url = "../data/iris.csv"
    data = pd.read_csv(url , delimiter=",", header=None, index_col=False)
    class_name,index = np.unique(data.iloc[:,-1],return_inverse=True)
    data.iloc[:,-1] = index
    data = data.loc[data[4] != 2]
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation

    model = Sequential()
    model.add(Dense(1, input_dim=4, activation='linear'))

    model.compile(loss='binary_crossentropy', optimizer={{choice(['adam', 'nadam'])}}, metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size={{choice([10, 40])}},
              epochs={{choice([100, 170])}},
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

# if __name__ == '__main__':
best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials())
x_train, y_train, x_test, y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(x_test, y_test))
