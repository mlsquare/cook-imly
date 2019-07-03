from imly.base import registrar
from imly.wrappers import XgboostKerasClassifier

@registrar.register_model
class RandomForest(object):
	''' Add model params '''
	wrapper = XgboostKerasClassifier
	module_name = 'xgboost'
	# self.model_params = None
	def create_model(self):
		from keras.models import Sequential
		from keras.layers.core import Dense

		# model_params = self.model_params
		model = Sequential()
		model.add(Dense(1, input_dim=2, activation='linear'))
		model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
		return model

	def set_params(self, params):
		pass

	def get_params(self):
		pass