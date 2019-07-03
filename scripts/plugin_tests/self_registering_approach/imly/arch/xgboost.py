__name__ = 'imly'
__package__ = 'imly'

from .base import registrar
from .wrappers import SklearnKerasClassifier, XgboostKerasClassifier

@registrar.register_model
class RandomForest(object):
	''' Add model params '''
	wrapper = XgboostKerasClassifier
	module_name = 'xgboost'
	# self.model_params = None
	def create_model(self):
		pass

	def set_params(self, params):
		pass

	def get_params(self):
		pass