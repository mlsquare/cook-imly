# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364. (https://arxiv.org/abs/1708.05027)
"""
import tensorflow as tf

#from tensorflow.keras.layers import Dense

from ..inputs import build_input_features, input_from_feature_columns
#from ..layers.core import PredictionLayer
from ..layers.interaction import FM
from ..layers.utils import concat_fun


def SVD(feature_columns, embedding_size=100,
        l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, bi_dropout=0,
        dnn_dropout=0):#, act_func='sigmoid', task='binary'):
    """Instantiates the Neural Factorization Machine architecture.

    :param feature_columns: An iterable containing all the sparse features used by model.
    :param num_factors: number of units in latent representation layer.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part.
    :param l2_reg_dnn: float . L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param biout_dropout: When not ``None``, the probability we will drop out the output of BiInteractionPooling Layer.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param act_func: Activation function to use at prediction layer.
    :param task: str, ``"binary"`` for  'binary_crossentropy' loss or  ``"multiclass"`` for 'categorical_crossentropy' loss
    :return: A Keras model instance.
    """
    #ensure that the `feature columns` is a list of `DenseFeat` Instances otherwise the model resulting here will have an Input shape (None,1)
    features = build_input_features(feature_columns)

    input_layers = list(features.values())

    sparse_embedding_list, _ = input_from_feature_columns(features,feature_columns, embedding_size, l2_reg_embedding, init_std, seed)
    
    fm_input = concat_fun(sparse_embedding_list, axis=1)
    fm_logit = FM()(fm_input)

    #if task=='binary':
    #    act_func = 'sigmoid'
    #    n_last = 1
    #elif task=='multiclass':
    #    act_func= 'softmax'
    #    n_last = 5

    #predictions = Dense(n_last, activation=act_func)(merge_layer)
    
    model = tf.keras.models.Model(inputs=input_layers, outputs=fm_logit)
    
    return model