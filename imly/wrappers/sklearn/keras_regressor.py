from keras.wrappers.scikit_learn import KerasRegressor
from optimizers.tune.tune import get_best_model
import pickle, onnxmltools


class SklearnKerasRegressor(KerasRegressor):
        def __init__(self, build_fn, **kwargs):
            super(KerasRegressor, self).__init__(build_fn=build_fn)
            self.primal = kwargs['primal']
            self.params = kwargs['params']

        def fit(self, x_train, y_train, **kwargs):
            kwargs.setdefault('params', self.params) # This params is to hold the values passed by the user
            kwargs.setdefault('space', False)
            primal_model = self.primal
            primal_model.fit(x_train, y_train)
            y_pred = primal_model.predict(x_train)
            primal_data = {
                'y_pred': y_pred,
                'model_name': primal_model.__class__.__name__
            }
            hyperopt_space = kwargs['space']
            self.params.update(kwargs['params']) # Merging params passed by user(if any) to the default params            

            # Search for best model using Tune
            self.model = get_best_model(x_train, y_train,
                                        primal_data=primal_data,
                                        params=self.params, space=hyperopt_space)
            self.model.fit(x_train, y_train, epochs=200,
                           batch_size=30, verbose=0)

            # TODO
            # Add a validation to check if the user has opted for 
            # optimization. If not, call 'fit' from KerasClassifier.

        def score(self, x, y, **kwargs):
            score = super(SklearnKerasRegressor, self).score(x, y, **kwargs)
            # keras_regressor treats all score values as loss and adds a '-ve' before passing
            return -score

        def save(self, using='dnn'):
            if using == 'sklearn':
                filename = 'scikit_model'
                pickle.dump(self.model, open(filename, 'wb'))
            else:
                onnx_model = onnxmltools.convert_keras(self.model)
                return onnx_model
