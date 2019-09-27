# from .arch.sklearn import registrar  ## Should be accessed via registry
from .base import registrar
from .arch.sklearn import LogisticRegression
# External file test -- Out of IMLY context
# from .plugins import external_plugins

def dope(primal_model, abstract_model=None, wrapper=None):
    ''' Import wrapper '''
    # print(LogisticRegression)
    module = primal_model.__class__.__module__.split('.')[0]
    model_name = primal_model.__class__.__name__

    ## Extract module and model names using functions from commons.
    # glm.register_plugin_dir('/home/shakkeel/Desktop/plugin_tests/selfregistering_approach/imly/plugins')

    ## Add an if-else to check the model version choice
    # abstract_model, wrapper = registrar[('sklearn', 'LogisticRegression')]['default'] # Change attrib name -- `models` to something else
    # abstract_model, wrapper = BaseModel.models[('sklearn', 'LogisticRegression')]


    model = wrapper(abstract_model=abstract_model) ## Get rid of build_fn
    # abstract_model = glm.plugins['LinearDiscrimantAnalysis']
    # print(glm.plugins)
    # model = SklearnKerasClassifier(build_fn=abstract_model.create_model())
    return model
    # return modellib.models


'''
Mapping process - 
1) glm.plugins will be {'SklearnLogisticRegression': [LogisticRegression, SklearnKerasClassifier]}
2) modellib.models will contain {('sklearn', 'LogisticRegression'): [LogisticRegression, SklearnKerasClassifier],
								('sklearn', 'LinearRegression'): [LinearRegression, SklearnKerasRegression]}

Updates - 

1) Make it 2 keys - Module name and model name
2) Rename and edit the structure of arch and base classes
3) Two key things - Single-point access and searchability
4) Add ABC methods
5) Do we need base class mixing at modellib level? What are the benefits in both cases?


Updates(28/06/2019)
1) Try registry as a decorator instead of class [X]
2) Remove build_fn [X]
3) Add Optimizer flag
4) @property decorator for 'wrapper' and 'model_name' varibales [X]
5) Error handling of model_params' keys
6) Extend this dummy arch closer to the real one
	- Add optimizer to the flow
	- Dry run with an actual model


Updates(01/07/2019)
1) Getting rid of registry class(by removing metaclass `Registrar`) [X]
2) model_dir changes - Trigger with @registry.register_model  [X]
3) Remove ModelStruct - Move to Registrar [X]
4) optimizer responses
5) Move code out of __init__ files
6) Why not load all arch/modules in dope level __init__ instead of find_modules?
	- Env/path variable issues
	- Dependancies
7) Benefits of find_models over __init__ import

Updates(02/07/2019)
1) Making externally declared models available in IMLY - Through dope
2) Mapping a given primal_model to multiple versions of it's dnn equivalent

'''