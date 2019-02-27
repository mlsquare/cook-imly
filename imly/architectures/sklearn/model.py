from numpy.random import seed
seed(3)
from tensorflow import set_random_seed
set_random_seed(3)

from keras.models import Sequential
from keras.layers.core import Dense
from utils.losses import mse_in_theano
from keras.regularizers import l2
import json, os


class create_model:
    def __init__(self, fn_name, param_name, **kwargs):
        self.fn_name = fn_name
        self.param_name = param_name
        self.x_train = None

    def __call__(self, **kwargs):
        try:
            module = __import__('architectures.sklearn.model',
                                fromlist=[self.fn_name]) 
            function = getattr(module, self.fn_name)
        except KeyError:
            print('Invalid model name passed to mapping_data')

        model = function(param_name=self.param_name,
                         x_train=kwargs['x_train'],
                         params=kwargs['params'])
        return model


def glm(**kwargs):

    model = Sequential()
    model.add(Dense(kwargs['params']['units'],
                    input_dim=kwargs['x_train'].shape[1],
                    activation=kwargs['params']['activation']))

    model.compile(optimizer=kwargs['params']['optimizer'],
                  loss=kwargs['params']['losses'],
                  metrics=['accuracy'])

    return model


def lda(**kwargs):

    model = Sequential()
    model.add(Dense(kwargs['params']['units'],
                    input_dim=kwargs['x_train'].shape[1],
                    activation=kwargs['params']['activation'][0],
                    kernel_regularizer=l2(1e-5)))
    model.compile(optimizer=kwargs['params']['optimizer'],
                #   loss=lda_loss(n_components=1, margin=1),
                  loss=mse_in_theano,
                  metrics=['accuracy'])
    # Metrics is usually provided through Talos.
    # Since we are bypassing Talos for LDA, we add the metrics directly.

    return model


# TODO
# reg_par missing for kernal_regularizer
