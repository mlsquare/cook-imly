# from ..base import Registrar
# from ..wrappers import SklearnKerasClassifier, XgboostKerasClassifier
# from abc import ABC, abstractmethod



# class modellib(metaclass=Registrar): ## Change to modellib.model ## model_registry
# 	pass
#  	# def __init__(self, model_class):
#  	# 	self.model_class = model_class


# '''Approach 1 - Single mount (Parametrized approach)'''
# # @register -- Registrar as the decorator
# # class BaseModel(Registrar, ABC):

# # 	@abstractmethod
# # 	def create_model(self):
# # 		raise NotImplementedError('Needs to be implemented!')

# # class MultiMeta(Registrar, ABC):
# # 	pass

# class glm():
# 	def create_model(self):
# 		from keras.models import Sequential
# 		from keras.layers.core import Dense

# 		model_params = self.model_params ## Validation of each key -- check keras search_kwargs() function in utils
# 		model = Sequential()
# 		model.add(Dense(model_params['units'], input_dim=model_params['input_dim'], activation=model_params['activation']))
# 		model.compile(optimizer=model_params['optimizer'], loss=model_params['loss'], metrics=['accuracy'])
# 		return model

# 	def set_params(self, params):
# 		self.model_params = params


# 	def get_params(self):
# 		return self.model_params

# ## @register -- instead of BaseModel or glm
# # class LogisticRegression(modellib, glm):
# # @modellib.register_model
# class LogisticRegression(modellib, glm):
# 	# __metaclass__ = modellib
# 	''' Add model params '''
# 	wrapper = SklearnKerasClassifier ## Okay to load at this level?
# 	module_name = 'sklearn' # Can be accessed directly from primal model
# 	## optimizer flag -- tune or hyperopt...
# 	def create_model(self):
# 		model_params = { 'units': 1,
# 						'input_dim': 2,
# 						'activation': 'sigmoid',
# 						'optimizer': 'adam',
# 						'loss': 'binary_crossentropy'
# 						}

# 		self.set_params(model_params)
# 		return super(LogisticRegression, self).create_model()


# class LinearRegression(modellib, glm):
# 	''' Add model params '''
# 	wrapper = SklearnKerasClassifier # Mandatory @property
# 	module_name = 'sklearn'
# 	def create_model(self):
# 		model_params = { 'units': 1,
# 						'input_dim': 2,
# 						'activation': 'linear',
# 						'optimizer': 'adam',
# 						'loss': 'mse'
# 						}

# 		self.set_params(model_params)
# 		return super(LinearRegression, self).create_model()

# class RandomForest(modellib):
# 	''' Add model params '''
# 	wrapper = XgboostKerasClassifier
# 	module_name = 'xgboost'
# 	# self.model_params = None
# 	def create_model(self):
# 		pass

# 	def set_params(self, params):
# 		pass

# 	def get_params(self):
# 		pass


# ----------------------- Approach 2 - Multiple metaclasses ------------------------------ #

# from ..base import Registrar
# from ..wrappers import SklearnKerasClassifier, XgboostKerasClassifier
# from abc import ABC, abstractmethod

# class BaseModelMeta(ABC):
#     def __init__(self, *args, **kwargs):
#         super(ConcreteClassMeta, self).__init__(*args, **kwargs)
#         if self.__abstractmethods__:
#             raise TypeError("{} has not implemented abstract methods {}".format(
#                 self.__name__, ", ".join(self.__abstractmethods__)))

# class modellib(Registrar, BaseModelMeta): ## Change to modellib.model ## model_registry
# 	pass


# class BaseModel(ABC):

# 	@abstractmethod
# 	def create_model(self):
# 		raise NotImplementedError('Needs to be implemented!')

# try:
# 	class glm(BaseModel):
# 		__metaclass__ = modellib
# 		def create_model(self):
# 			from keras.models import Sequential
# 			from keras.layers.core import Dense

# 			model_params = self.model_params ## Validation of each key -- check keras search_kwargs() function in utils
# 			model = Sequential()
# 			model.add(Dense(model_params['units'], input_dim=model_params['input_dim'], activation=model_params['activation']))
# 			model.compile(optimizer=model_params['optimizer'], loss=model_params['loss'], metrics=['accuracy'])
# 			return model

# 		def set_params(self, params):
# 			self.model_params = params


# 		def get_params(self):
# 			return self.model_params

# except TypeError as e:
#     print("Couldn't create class --", e)

# ## @register -- instead of BaseModel or glm
# # class LogisticRegression(modellib, glm):
# # @modellib.register_model
# class LogisticRegression(glm):
# 	__metaclass__ = modellib
# 	''' Add model params '''
# 	wrapper = SklearnKerasClassifier ## Okay to load at this level?
# 	module_name = 'sklearn' # Can be accessed directly from primal model
# 	## optimizer flag -- tune or hyperopt...
# 	def create_model(self):
# 		model_params = { 'units': 1,
# 						'input_dim': 2,
# 						'activation': 'sigmoid',
# 						'optimizer': 'adam',
# 						'loss': 'binary_crossentropy'
# 						}

# 		self.set_params(model_params)
# 		return super(LogisticRegression, self).create_model()


# ----------------------- Approach 3 - Class as decorator ------------------------------ #


# from ..base import Registrar, registry
# from ..wrappers import SklearnKerasClassifier, XgboostKerasClassifier
# from abc import ABC, abstractmethod



# class modellib(metaclass=Registrar): ## Change to modellib.model ## model_registry
# 	pass
# #  	# def __init__(self, model_class):
# #  	# 	self.model_class = model_class


# '''Approach 1 - Single mount (Parametrized approach)'''
# # @register -- Registrar as the decorator
# class BaseModel(ABC):

# 	@abstractmethod
# 	def create_model(self):
# 		raise NotImplementedError('Needs to be implemented!')

# 	@property
# 	@abstractmethod
# 	def wrapper(self):
# 		raise NotImplementedError('Needs to be implemented!')

# 	@wrapper.setter
# 	def wrapper(self, obj):
# 		self._wrapper = obj

# 	@abstractmethod
# 	def create_model(self):
# 		raise NotImplementedError('Needs to be implemented!')

# # class MultiMeta(Registrar, ABC):
# # 	pass

# class glm(BaseModel):
# 	def wrapper(self):
# 		return self._wrapper

# 	def create_model(self):
# 		from keras.models import Sequential
# 		from keras.layers.core import Dense

# 		model_params = self.model_params ## Validation of each key -- check keras search_kwargs() function in utils
# 		model = Sequential()
# 		model.add(Dense(model_params['units'], input_dim=model_params['input_dim'], activation=model_params['activation']))
# 		model.compile(optimizer=model_params['optimizer'], loss=model_params['loss'], metrics=['accuracy'])
# 		return model

# 	def set_params(self, params):
# 		self.model_params = params


# 	def get_params(self):
# 		return self.model_params

# ## @register -- instead of BaseModel or glm
# # class LogisticRegression(modellib, glm):
# @registry.register_model # Allocate to Registrar
# # @registry.register_model_dir -- dir should be detected internally
# class LogisticRegression(glm):
# 	# __metaclass__ = modellib
# 	''' Add model params '''
# 	def __init__(self):
# 		self.wrapper = SklearnKerasClassifier
# 		wrapper = self.wrapper
# 	# wrapper = SklearnKerasClassifier ## Okay to load at this level?
# 	module_name = 'sklearn' # Can be accessed directly from primal model
# 	## optimizer flag -- tune or hyperopt...
# 	def create_model(self):
# 		model_params = { 'units': 1,
# 						'input_dim': 2,
# 						'activation': 'sigmoid',
# 						'optimizer': 'adam',
# 						'loss': 'binary_crossentropy'
# 						}

# 		self.set_params(model_params)
# 		return super().create_model()

# @modellib.register_model
# class LinearRegression(glm):
# 	''' Add model params '''
# 	wrapper = SklearnKerasClassifier # Mandatory @property
# 	module_name = 'sklearn'
# 	def create_model(self):
# 		model_params = { 'units': 1,
# 						'input_dim': 2,
# 						'activation': 'linear',
# 						'optimizer': 'adam',
# 						'loss': 'mse'
# 						}

# 		self.set_params(model_params)
# 		return super(LinearRegression, self).create_model()

# class RandomForest(modellib):
# 	''' Add model params '''
# 	wrapper = XgboostKerasClassifier
# 	module_name = 'xgboost'
# 	# self.model_params = None
# 	def create_model(self):
# 		pass

# 	def set_params(self, params):
# 		pass

# 	def get_params(self):
# 		pass


# ----------------------- Approach 4 - Registrar as a class(not a metaclass) ------------------------------ #


# from ..base import registrar
__name__ = 'imly' ## Is this the right approach?
__package__ = 'imly'
print('name -- ', __name__)
print('package -- ', __package__)
from .base import registrar
from .wrappers import SklearnKerasClassifier
# from wrappers import SklearnKerasClassifier, XgboostKerasClassifier
# import registrar
from abc import ABC, abstractmethod
import os

class BaseModel(ABC):

	@abstractmethod
	def create_model(self):
		raise NotImplementedError('Needs to be implemented!')

	@property
	@abstractmethod
	def wrapper(self):
		raise NotImplementedError('Needs to be implemented!')

	@wrapper.setter
	def wrapper(self, obj):
		self._wrapper = obj

	@abstractmethod
	def create_model(self):
		raise NotImplementedError('Needs to be implemented!')

class glm(BaseModel):
	def wrapper(self):
		return self._wrapper

	def create_model(self):
		from keras.models import Sequential
		from keras.layers.core import Dense

		model_params = self.model_params ## Validation of each key -- check keras search_kwargs() function in utils
		model = Sequential()
		model.add(Dense(model_params['units'], input_dim=model_params['input_dim'], activation=model_params['activation']))
		model.compile(optimizer=model_params['optimizer'], loss=model_params['loss'], metrics=['accuracy'])
		return model

	def set_params(self, params):
		self.model_params = params


	def get_params(self):
		return self.model_params


@registrar.register_model # Allocate to Registrar
# @registry.register_model_dir -- dir should be detected internally
class LogisticRegression(glm):
	''' Add model params '''
	def __init__(self):
		self.wrapper = SklearnKerasClassifier
		wrapper = self.wrapper
	module_name = 'sklearn' # Can be accessed directly from primal model
	## optimizer flag -- tune or hyperopt...
	model_dir = os.path.dirname(os.path.realpath(__file__)) # This can come from the base class
	def create_model(self):
		model_params = { 'units': 1,
						'input_dim': 2,
						'activation': 'sigmoid',
						'optimizer': 'adam',
						'loss': 'binary_crossentropy'
						}

		self.set_params(model_params)
		return super().create_model()

@registrar.register_model
class LinearRegression(glm):
	''' Add model params '''
	wrapper = SklearnKerasClassifier # Mandatory @property
	module_name = 'sklearn'
	def create_model(self):
		model_params = { 'units': 1,
						'input_dim': 2,
						'activation': 'linear',
						'optimizer': 'adam',
						'loss': 'mse'
						}

		self.set_params(model_params)
		print(LinearRegression)
		return super().create_model()

