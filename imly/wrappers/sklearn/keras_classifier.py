from keras.wrappers.scikit_learn import KerasClassifier
from utils.model_mapping import get_model_design
from architectures.sklearn.model import create_model
from optimizers.talos.talos import get_best_model
import pickle, onnxmltools
import numpy as np


class SklearnKerasClassifier(KerasClassifier):
        def __init__(self, build_fn, **kwargs):
            super(KerasClassifier, self).__init__(build_fn=build_fn)
            self.primal = kwargs['primal']
            self.params = kwargs['params']
            self.val_metric = kwargs['val_metric']
            self.metric = kwargs['metric']

        def fit(self, x_train, y_train, **kwargs):
            print('Keras classifier chosen')
            primal_model = self.primal
            primal_model.fit(x_train, y_train)
            y_pred = primal_model.predict(x_train)
            primal_data = {
                'y_pred': y_pred,
                'model_name': primal_model.__class__.__name__
            }

            # This is to update the 'classes_' variable used in keras_regressor
            y_train = np.array(y_train)
            if len(y_train.shape) == 2 and y_train.shape[1] > 1:
                self.classes_ = np.arange(y_train.shape[1])
            elif (len(y_train.shape) == 2 and y_train.shape[1] == 1) or len(y_train.shape) == 1:
                self.classes_ = np.unique(y_train)
                y_train = np.searchsorted(self.classes_, y_train)
            else:
                raise ValueError('Invalid shape for y_train: ' + str(y_train.shape))

            if (primal_model.__class__.__name__ != 'LinearDiscriminantAnalysis'):
                self.model, final_epoch, final_batch_size = get_best_model(x_train, y_train,
                                                                        primal_data=primal_data,
                                                                        params=self.params, 
                                                                        val_metric=self.val_metric, 
                                                                        metric=self.metric) 
                self.model.fit(x_train, y_train, epochs=final_epoch,
                               batch_size=final_batch_size, verbose=0)

            # Temporarily bypassing Talos optimization for LDA
            else:
                fn_name, param_name = get_model_design(primal_data['model_name'])
                mapping_instance = create_model(fn_name=fn_name, param_name=param_name)
                self.model = mapping_instance.__call__(x_train=x_train)
                self.model.fit(x_train, y_train, epochs=500, batch_size=100)

            final_model = self.model
            return final_model

        def save(self, using='dnn'):
            if using == 'sklearn':
                filename = 'scikit_model'
                pickle.dump(self.model, open(filename, 'wb'))
            else:
                onnx_model = onnxmltools.convert_keras(self.model)
                return onnx_model

        # def predict_classes(self, x):
        #     predicted_classes = self.model.predict_classes(x)
        #     print(predicted_classes)
        #     return predicted_classes
