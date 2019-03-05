import onnxmltools
import pickle
import numpy as np
from optimizers.tune.tune import get_best_model
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import OneHotEncoder
# from tensorflow import set_random_seed
# from numpy.random import seed


class SklearnKerasClassifier(KerasClassifier):
    def __init__(self, build_fn, **kwargs):
        super(KerasClassifier, self).__init__(build_fn=build_fn)
        self.primal = kwargs['primal']
        self.params = kwargs['params']
        self.encoder = OneHotEncoder(handle_unknown='error')
        self.classification_type = 'binary'

    def fit(self, x_train, y_train, **kwargs):
        print('Keras classifier chosen')

        # This params is to hold the values passed by the user
        kwargs.setdefault('params', self.params)
        kwargs.setdefault('space', False)
        primal_model = self.primal
        primal_model.fit(x_train, y_train)
        y_pred = primal_model.predict(x_train)
        primal_model_name = primal_model.__class__.__name__

        # Update class_name with 'Multiclass'
        # class_name is used to retrieve the model-param mapping
        # model-param mapping available at imly/utils/model_params_mapping.json
        if primal_model.classes_.shape[0] != 2:
            self.classification_type = 'multiclass'
            primal_model_name = primal_model.__class__.__name__ + 'MultiClass'
            '''
            Notes on encoding -
            1) y_train gets one_hot_encoded inorder for it to be
            compatible with it's corresponding model.
            2) The same encoder instance is used to encode y_pred and 
            y_test(in score method).
            '''
            self.encoder.fit(y_train)
            y_train = self.encoder.transform(y_train)
            y_pred = self.encoder.transform(y_pred.reshape(-1, 1))
            print(primal_model_name, " --- from keras_classifier.py")

        primal_data = {
            'y_pred': y_pred,
            'model_name': primal_model_name
        }
        hyperopt_space = kwargs['space']

        # Merging params passed by user(if any) to the default params
        self.params.update(kwargs['params'])

        '''
        Note -
        This is to update the 'classes_' variable used in keras_regressor.
        'classes_' variable is used by the score function of keras_regressor.
        An alternate option would be to create our own score function.
        Move this to a wrapper score.
        '''

        # y_train_temp = np.array(y_train)
        # if len(y_train_temp.shape) == 2 and y_train_temp.shape[1] > 1:
        #     self.classes_ = np.arange(y_train_temp.shape[1])
        # elif (len(y_train_temp.shape) == 2 and y_train_temp.shape[1] == 1) or len(y_train_temp.shape) == 1:
        #     self.classes_ = np.unique(y_train_temp)
        #     y_train_temp = np.searchsorted(self.classes_, y_train_temp)
        # else:
        #     raise ValueError(
        #         'Invalid shape for y_train_temp: ' + str(y_train_temp.shape))

        if (kwargs['params'] != self.params or kwargs['space']):
            ## Search for best model using Tune ##
            self.model = get_best_model(x_train, y_pred,
                                        primal_data=primal_data,
                                        params=self.params, space=hyperopt_space)
            # self.model.fit(x_train, y_train, epochs=200,
            #                batch_size=30, verbose=0)
        else:
            # This else case is triggred if the user opts out of optimization
            mapping_instance = self.build_fn
            # build_fn passed from dope holds the class instance with param_name and fn_name.
            # Hence, mapping variables are already available.
            self.model = mapping_instance.__call__(x_train=x_train,
                                                   y_train=y_pred,
                                                   params=kwargs['params'])
            self.model.fit(x_train, y_pred)

        # TODO
        # Add a validation to check if the user has opted for
        # optimization. If not, call 'fit' from KerasClassifier.

    def save(self, using='dnn'):
        if using == 'sklearn':
            filename = 'scikit_model'
            pickle.dump(self.model, open(filename, 'wb'))
        else:
            onnx_model = onnxmltools.convert_keras(self.model)
            return onnx_model

    def explain(self):
        return self.model.summary()

    def score(self, x_test, y_test):
        # TODO 
        # 1) Raise error if y_test contains unknown
        # labels. IMP
        # 2) Cross check this implementation on Binary classification
        # 3) Transform if multiclass
        # 4) Cross check the value returned by evaluate
        if self.classification_type == 'multiclass':
            try:
                y_test = self.encoder.transform(y_test)
            except ValueError as error:
                # print(error)
                print("This usually happens if your test_train_split is not stratified.\
                Try using a stratified test_train_split.")
                raise error

        score = self.model.evaluate(x_test, y_test)
        print(score)
        return score[1]
