from keras.wrappers.scikit_learn import KerasClassifier

class SklearnKerasClassifier(KerasClassifier):
    def __init__(self, **kwargs):
        model = kwargs['abstract_model']
        super(KerasClassifier, self).__init__(build_fn=model.create_model())

    def fit(self, X, y, **kwargs):
        pass