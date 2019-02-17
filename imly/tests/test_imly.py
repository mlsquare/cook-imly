from . import automation_script
from os import path
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler


'''
The automation_script has been altered in such a way that it returns
a dict called 'exp_results'. 
The structure of exp_results is as follows -

    exp_results = {
        'keras': keras_score,
        'scikit': primal_score,
        'kfold': None,
        'fig_url': fig_url,
        'correlation': correlation_value
    }

test_classification() and test_regression asserts that the accuracy values
are in between 0 and 1.

'''


def load_diabetes():

    dataset_info = automation_script.get_dataset_info("diabetes")
    url = "../data/diabetes.csv"
    data = pd.read_csv(url, delimiter=",", header=None, index_col=False)
    sc = StandardScaler()
    data = sc.fit_transform(data)
    data = pd.DataFrame(data)

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    # X = preprocessing.scale(X)
    # Y = preprocessing.normalize(Y)

    exp_results = automation_script.run_imly(dataset_info, 'linear_regression', X, Y, 0.60, return_exp_results=True)
    return exp_results


def load_iris():
    dataset_name = "uci_iris"
    dataset_info = automation_script.get_dataset_info(dataset_name)

    url = "../data/iris.csv"
    data = pd.read_csv(url, delimiter=",", header=None, index_col=False)
    class_name, index = np.unique(data.iloc[:,-1],return_inverse=True)
    data.iloc[:, -1] = index
    data = data.loc[data[4] != 2]
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    exp_results = automation_script.run_imly(dataset_info, 'logistic_regression', X, Y, 0.60, return_exp_results=True)
    return exp_results


def load_salary_dataset():
    dataset_name = "uci_adult_salary"
    dataset_info = automation_script.get_dataset_info(dataset_name)

    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
            'hours-per-week', 'native-country', 'target']
    url = "../data/iris.csv" if path.exists("../data/dataset.csv.csv") else dataset_info['url']
    data = pd.read_csv(url, delimiter=" ", header=None, names=names)

    data = data[data["workclass"] != "?"]
    data = data[data["occupation"] != "?"]
    data = data[data["native-country"] != "?"]

    # Convert categorical fields #
    categorical_col = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'native-country', 'target']

    for col in categorical_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c

    feature_list = names[:14]
    # Test train split #
    X = data.loc[:, feature_list]
    Y = data[['target']]
    exp_results = automation_script.run_imly(dataset_info, 'logistic_regression', X, Y, 0.60, return_exp_results=True)
    return exp_results


def load_airfoil():
    dataset_name = "uci_airfoil"
    dataset_info = automation_script.get_dataset_info(dataset_name)

    data = pd.read_csv("../data/uci_airfoil_self_noise.csv", delimiter=",", header=0, index_col=0)
    sc = StandardScaler()
    data = sc.fit_transform(data)
    data = pd.DataFrame(data)

    Y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    exp_results = automation_script.run_imly(dataset_info, 'logistic_regression', X, Y, 0.60, return_exp_results=True)
    return exp_results


def test_classification():
    iris_results = load_iris()
    assert 0 <= iris_results['keras'] <= 1
    assert 0 <= iris_results['scikit'] <= 1

    salary_results = load_salary_dataset()
    assert 0 <= salary_results['keras'] <= 1
    assert 0 <= salary_results['scikit'] <= 1

    airfoil_results = load_airfoil()
    assert 0 <= airfoil_results['keras'] <= 1
    assert 0 <= airfoil_results['scikit'] <= 1


def test_regression():
    diabetes_results = load_diabetes()
    assert 0 <= diabetes_results['keras'] <= 1
    assert 0 <= diabetes_results['scikit'] <= 1
    assert diabetes_results['correlation'] >= 0.5
