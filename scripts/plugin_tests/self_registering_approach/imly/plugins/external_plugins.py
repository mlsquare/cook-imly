from ..arch.sklearn import modellib, glm

class LinearDiscrimantAnalysis(modellib, glm):
	wrapper = SklearnKerasClassifier
	module_name = 'sklearn'
	''' Add model params '''
	def set_params(self):
		pass

	def get_params(self):
		pass