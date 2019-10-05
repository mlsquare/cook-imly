# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364. (https://arxiv.org/abs/1708.05027)
"""
import tensorflow as tf

#from tensorflow.python.keras.layers import Dense
from ..inputs import build_input_features, input_from_feature_columns

from ..layers.interaction import FM
from ..layers.utils import concat_fun

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.inputs import SparseFeat
#from deepctr.inputs import DenseFeat#SparseFeat #Avoid using SparseFeat for Model Input shape to be (None, feature_column.dimension) not fixed (None,1)

#from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

class SVD_C:
    def __init__(self, feat_cols):
        self.feature_columns = feat_cols

    
    @classmethod
    def prepare_data(cls, path, sparse_features, task='binary'):
        data_path = path
        dataframe = pd.read_csv(data_path, names= 'user_id,movie_id,rating,timestamp'.split(','))
        sparse_features = sparse_features# ["movie_id", "user_id"]
        y= ['rating']
        
        for feat in sparse_features:
            lbe = LabelEncoder()
            dataframe[feat] = lbe.fit_transform(dataframe[feat])

        #feature_columns = [DenseFeat(feat, dataframe[feat].nunique()) for feat in sparse_features]
        feature_columns = [SparseFeat(feat, dataframe[feat].nunique()) for feat in sparse_features]

        trainset, testset = train_test_split(dataframe, test_size=0.2)

        #train_model_input = [to_categorical(trainset[fc.name].values, num_classes= fc.dimension) for fc in feature_columns]#includes values from only data[user_id], data[movie_id]
        #test_model_input = [to_categorical(testset[fc.name].values, num_classes= fc.dimension) for fc in feature_columns]#includes values from only data[user_id], data[movie_id]
        train_model_input = [trainset[name].values for name in sparse_features]#includes values from only data[user_id], data[movie_id]
        test_model_input = [testset[name].values for name in sparse_features]#includes values from only data[user_id], data[movie_id]

        
        if task =='binary':
            train_lbl = trainset[y].values
            test_lbl= testset[y].values

        #elif task == 'multiclass':
        #    train_lbl = to_categorical(trainset[y])[:,1:]#stripping 0th column as rating is (1,5)
        #    test_lbl= to_categorical(testset[y])[:,1:]#stripping 0th column as rating is (1,5)
        #else:
        #    raise ValueError("Enter task either 'binary' or 'multiclass'")
        
        return cls(feature_columns), (train_model_input, train_lbl), (test_model_input, test_lbl) #try returning train_model_input from inside __init__()

    def create_model(self, embedding_size=100,
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
        features = build_input_features(self.feature_columns)

        input_layers = list(features.values())

        sparse_embedding_list, _ = input_from_feature_columns(features,feature_columns, embedding_size, l2_reg_embedding, init_std, seed)
    
        fm_input = concat_fun(sparse_embedding_list, axis=1)
        fm_logit = FM()(fm_input)


        #hid_layer_1= Dense(num_factors)(input_layers[0])
        #hid_layer_2= Dense(num_factors)(input_layers[1])
        
        #merge_layer = tf.keras.layers.dot([hid_layer_1, hid_layer_2], axes=1)
        #if task=='binary':
        #    act_func = 'sigmoid'
        #    n_last = 1
        #elif task=='multiclass':
        #    act_func= 'softmax'
        #    n_last = 5
        #else:
        #    raise ValueError("Enter task either 'binary' or 'multiclass'")

        #predictions = Dense(n_last, activation=act_func)(merge_layer)
        
        model = tf.keras.models.Model(inputs=input_layers, outputs=fm_logit)
        
        return model