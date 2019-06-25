import base
from base import registry
from keras.models import Sequential
from keras.layers.core import Dense


@registry.register
class glm(base.Model):

	def __init__(self, config):
		self.config = config


	def get_model(self):
		model = Sequential()
		model.add(Dense(units=1, input_dim=2, activation='linear'))
		model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
		return model


# __all__ = list(registry.keys())