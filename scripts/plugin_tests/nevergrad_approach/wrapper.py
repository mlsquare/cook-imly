from architecture import glm
from configurations import model_param
from base import registry

# dynamic_model = glm(model_param)

# print('From architecture.py -- ', dynamic_model)
print('From registry -- ', registry['glm'](model_param))
print(sorted(registry.keys()))