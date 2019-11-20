#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..optmizers import get_best_model
from ..utils.functions import _parse_params
import pickle
import onnxmltools
import numpy as np

import tensorflow as tf
from keras.utils import to_categorical

class SklearnKerasDecompose():
    def __init__(self, proxy_model, primal_model, **kwargs):
        self.primal_model = primal_model
        self.params = None ## Temporary!
        self.proxy_model = proxy_model
        
    def fit(self, X, y=None, **kwargs):
        self.fit_transform(X)
        return self
    
    def fit_transform(self, X, y=None,**kwargs):
        kwargs.setdefault('full_matrices', False)
        kwargs.setdefault('params', self.params)
        kwargs.setdefault('space', False)
        kwargs.setdefault('compute_uv', True)
        kwargs.setdefault('name', None)
        self.params = kwargs['params']
        X = np.array(X)
        y = np.array(y)
        
        primal_model = self.primal_model
        self.proxy_model.num_components= primal_model.n_components
        
        sess= tf.Session()#for TF  1.13
        s,u,v= sess.run(tf.linalg.svd(X))#for TF  1.13
        #s: singular values
        #u: normalised projection distances
        #v: decomposition/projection orthogonal axes
        
        self.proxy_model.components_= v[:self.proxy_model.num_components,:]#analogous to TruncatedSVD().components_ Or primal_model.components_ Or Vh component from randomised SVD
        #Sigma = s[:self.proxy_model.num_components]
        X_transformed = u[:,:self.proxy_model.num_components] * s[:self.proxy_model.num_components]
        self.singular_values_ = s[:self.proxy_model.num_components]# Store the n_components singular values            
        return X_transformed
    
    def explained_variance_(self):
        print('Method not implemented yet!')
        
    def explained_variance_ratio_(self):
        print('Method not implemented yet!')    
    
    def explain(self, **kwargs):
        # @param: SHAP or interpret
        print('Coming soon...')

    
class SklearnKerasDecompose_1():
    def __init__(self, proxy_model, primal_model, **kwargs):
        self.primal_model = primal_model
        self.params = None ## Temporary!
        self.proxy_model = proxy_model
        
    def fit(self, X, y=None, **kwargs):
        self.fit_transform(X)
        return self
    
    def fit_transform(self, X, y=None,**kwargs):
        pass
    
    def pseudo_fit(self, X, y, **kwargs):
        self.pseudo_fit_transform(X)
        return self
    
    def pseudo_fit_transform(self, X, y,**kwargs):
        kwargs.setdefault('verbose', 1)
        kwargs.setdefault('params', self.params)
        kwargs.setdefault('space', False)
        kwargs.setdefault('epochs', 64)
        kwargs.setdefault('batch_size', 32)
        self.params = kwargs['params']
        X = np.array(X)
        y = np.array(y)

        primal_model = self.primal_model
        self.pronum_components= primal_model.n_components
        
        primal_model.fit(X)
        y_pred = primal_model.transform(X)

        #X, y, y_pred = self.proxy_model.transform_data(X, y, y_pred)

        # This should happen only after transformation.        

        #try:
        #    self.proxy_model.in_columns = list(X.columns)
         #   X = np.array(X)
          #  y = np.array(y)     
            
        self.proxy_model.X = X ##  abstract -> model_skeleton
        self.proxy_model.y = y
        self.proxy_model.primal = self.primal_model
        
        primal_model = self.primal_model
        

        model = self.proxy_model.create_model()
        
        input_list = []
        
        for idx in range(X.shape[1]):
            input_list.append(to_categorical(X[:,idx], num_classes=np.unique(X[:,idx]).shape[0]))
        
        
        self.final_model = model.fit(input_list, y_pred, epochs=kwargs['epochs'],
                                     batch_size=kwargs['batch_size'],
                                     verbose=kwargs['verbose'])
        
        return self.final_model
    
class SklearnKerasClassifier():
    """
	Adapter to connect sklearn classifier algorithms with keras models.

    This class can be used as an adapter for any primal classifier that relies
    on keras as the backend for proxy model.

    Parameters
    ----------
    proxy_model : proxy model instance
        The proxy model passed from dope.

    primal_model : primal model instance
        The primal model passed from dope.

    params : dict, optional
        Additional model params passed by the user.


    Methods
    -------
	fit(X, y)
        Method to train a transpiled model

	save(filename)
        Method to save a trained model. This method saves
        the models in three formals -- pickle, h5 and onnx.
        Expects 'filename' as a string.

	score(X, y)
        Method to score a trained model.

	predict(X)
        This method returns the predicted values for a
        trained model.

	explain()
        Method to provide model interpretations(Yet to be implemented)

    """

    def __init__(self, proxy_model, primal_model, **kwargs):
        self.primal_model = primal_model
        self.params = None ## Temporary!
        self.proxy_model = proxy_model

    def fit(self, X, y, **kwargs):
        kwargs.setdefault('cuts_per_feature', None) ## Better way to handle?

        self.proxy_model.cuts_per_feature = kwargs['cuts_per_feature'] ## For all models?
        kwargs.setdefault('verbose', 0)
        kwargs.setdefault('params', self.params)
        kwargs.setdefault('space', False)
        kwargs.setdefault('epochs', 250)
        kwargs.setdefault('batch_size', 30)
        self.params = kwargs['params']
        X = np.array(X)
        y = np.array(y)

        primal_model = self.primal_model
        primal_model.fit(X, y)
        y_pred = primal_model.predict(X)

        X, y, y_pred = self.proxy_model.transform_data(X, y, y_pred)

        # This should happen only after transformation.
        self.proxy_model.X = X ##  abstract -> model_skeleton
        self.proxy_model.y = y
        self.proxy_model.primal = self.primal_model

        if self.params != None: ## Validate implementation with different types of tune input
            if not isinstance(self.params, dict):
                raise TypeError("Params should be of type 'dict'")
            self.params = _parse_params(self.params, return_as='flat')
            self.proxy_model.update_params(self.params)

        primal_data = { ## Consider renaming -- primal_model_data or primal_results
            'y_pred': y_pred,
            'model_name': primal_model.__class__.__name__
        }

        ## Search for best model using Tune ##
        self.final_model = get_best_model(X, y, proxy_model = self.proxy_model,
                                            primal_data=primal_data, epochs=kwargs['epochs'], batch_size=kwargs['batch_size'],
                                            verbose=kwargs['verbose'])
        return self.final_model  # Return self? IMPORTANT

    def save(self, filename=None):
        if filename == None:
            raise ValueError(
                'Name Error: to save the model you need to specify the filename')

        pickle.dump(self.final_model, open(filename + '.pkl', 'wb'))

        self.final_model.save(filename + '.h5')

        onnx_model = onnxmltools.convert_keras(self.final_model)
        onnxmltools.utils.save_model(onnx_model, filename + '.onnx')

    def score(self, X, y, **kwargs):
        if self.proxy_model.enc is not None:
            ## Should we accept pandas?
            y = np.array(y)
            X = np.array(X)
            if len(y.shape) == 1 or y.shape[1] == 1:
                y = self.proxy_model.enc.transform(y.reshape(-1,1))
                y = y.toarray() ## Cross check with logistic regression flow
            else:
                y = self.proxy_model.enc.transform(y)
                y = y.toarray()
        score = self.final_model.evaluate(X, y, **kwargs)
        return score

    def predict(self, X):
        X = np.array(X)
        if hasattr(self.final_model, 'predict_classes'):
            pred = self.final_model.predict_classes(X)
        else:
            pred = self.final_model.predict(X)
            pred = np.argmax(pred,axis=1)
        return pred

    def predict_proba(self, X):
        pass

    def explain(self, **kwargs):
        # @param: SHAP or interpret
        print('Coming soon...')
        return self.final_model.summary()



class SklearnKerasRegressor():
    """
	Adapter to connect sklearn regressor algorithms with keras models.

    This class can be used as an adapter for any primal regressor that relies
    on keras as the backend for proxy model.

    Parameters
    ----------
    proxy_model : proxy model instance
        The proxy model passed from dope.

    primal_model : primal model instance
        The primal model passed from dope.

    params : dict, optional
        Additional model params passed by the user.


    Methods
    -------
	fit(X, y)
        Method to train a transpiled model

	save(filename)
        Method to save a trained model. This method saves
        the models in three formals -- pickle, h5 and onnx.
        Expects 'filename' as a string.

	score(X, y)
        Method to score a trained model.

	predict(X)
        This method returns the predicted values for a
        trained model.

	explain()
        Method to provide model interpretations(Yet to be implemented)

    """

    def __init__(self, proxy_model, primal_model, **kwargs):
        self.primal_model = primal_model
        self.proxy_model = proxy_model
        self.params = None

    def fit(self, X, y, **kwargs):
        self.proxy_model.X = X
        self.proxy_model.y = y
        self.proxy_model.primal = self.primal_model
        kwargs.setdefault('verbose', 0)
        kwargs.setdefault('epochs', 250)
        kwargs.setdefault('batch_size', 30)
        kwargs.setdefault('params', self.params)
        self.params = kwargs['params']

        if self.params != None: ## Validate implementation with different types of tune input
            if not isinstance(self.params, dict):
                raise TypeError("Params should be of type 'dict'")
            self.params = _parse_params(self.params, return_as='flat')
            self.proxy_model.update_params(self.params)
        primal_model = self.primal_model
        primal_model.fit(X, y)
        y_pred = primal_model.predict(X)
        primal_data = {
            'y_pred': y_pred,
            'model_name': primal_model.__class__.__name__
        }

        self.final_model = get_best_model(X, y, proxy_model=self.proxy_model, primal_data=primal_data,
                                          epochs=kwargs['epochs'], batch_size=kwargs['batch_size'],
                                          verbose=kwargs['verbose'])
        return self.final_model  # Not necessary.

    def score(self, X, y, **kwargs):
        score = self.final_model.evaluate(X, y, **kwargs)
        return score

    def predict(self, X):
        '''
        Pending:
        1) Write a 'filter_sk_params' function(check keras_regressor wrapper) if necessary.
        2) Data checks and data conversions
        '''
        pred = self.final_model.predict(X)
        return pred

    def save(self, filename=None):
        if filename == None:
            raise ValueError(
                'Name Error: to save the model you need to specify the filename')

        pickle.dump(self.final_model, open(filename + '.pkl', 'wb'))

        self.final_model.save(filename + '.h5')

        onnx_model = onnxmltools.convert_keras(self.final_model)
        onnxmltools.utils.save_model(onnx_model, filename + '.onnx')

    def explain(self, **kwargs):
        # @param: SHAP or interpret
        print('Coming soon...')
        return self.final_model.summary()

class SklearnPytorchClassifier():
    def __init__(self, proxy_model, primal_model, **kwargs):
        self.primal_model = primal_model
        self.params = None ## Temporary!
        self.proxy_model = proxy_model

    def fit(self, X, y, **kwargs):
        self.proxy_model.X = X
        self.proxy_model.y = y
        self.proxy_model.primal = self.primal_model

        for epoch in range(50):
            # Forward Propagation
            # Access model, criterion and optimizer from proxy_model
            # Alter how tune computes `fit`. Override keras_model.fit option
            y_pred = model(x)    # Compute and print loss
            loss = criterion(y_pred, y)
            print('epoch: ', epoch,' loss: ', loss.item())    # Zero the gradients
            optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            optimizer.step()


# TODO
# predict_proba implementation
# filter_sklearn_params method