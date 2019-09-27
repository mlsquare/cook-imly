import os, imp
from abc import ABC, abstractmethod


# ---------------- Approach 1 - Registrar as a metaclass ----------------

# class Registrar(type): ## Registrar
# class Registrar(type):
# class Registrar():
#     ''' This class acts as a mount point for our plugins    '''

#     # Default path to search for plugins - change with register_plugin_dir
#     model_collection_path = os.path.join(__file__, 'models') 

#     def __init__(self, name, bases, attrs):
#         ''' Initializing mount, or registering a plugin?    '''
#         if not hasattr(self, 'models'):
#             print('Mount initialized')
#             # self.models = ModelStruct(Registrar)
#             self.models = {}
#         else:
#             print('Model registered')
#             self.register_model(self)
    
#     def register_model(self, model):
#         ''' Registration logic + append to plugins struct '''
#         print(model)
#         # model = model(model.__class__.__name__) #< Init the plugin
#         model = model()
#         # self.models[model.__class__.__name__] = model

#         wrapper = model.wrapper
#         module_name = model.module_name
#         self.models[(module_name, model.__class__.__name__)] = [model, wrapper]
    
#     @staticmethod
#     def register_model_dir(model_collection_path):
#         ''' This function sets the plugin path to be searched   '''
#         if os.path.isdir(model_collection_path):
#             Registrar.model_collection_path = model_collection_path
#         else:
#             raise EnvironmentError('%s is not a directory' % model_collection_path)
        
#     @staticmethod
#     def find_models(): # Change name to `load...`
#         ''' Traverse registered plugin directory and import non-loaded modules  '''
#         model_collection_path = Registrar.model_collection_path
#         if not os.path.isdir(model_collection_path):
#             raise EnvironmentError('%s is not a directory' % model_collection_path)
        
#         for file_ in os.listdir(model_collection_path):
#             if file_.endswith('.py') and file_ != '__init__.py':
#                 module = file_[:-3] #< Strip extension
#                 mod_obj = globals().get(module)
#                 if mod_obj is None:
#                     f, filename, desc = imp.find_module(
#                         module, [model_collection_path])
#                     globals()[module] = mod_obj = imp.load_module(
#                         module, f, filename, desc)


# ---------------- Approach 2 - Registrar as a class ----------------

# Registrar to Registry?
# Change models to registry

class Registrar(object):
    ''' This class acts as a mount point for our plugins    '''

    # Default path to search for plugins - change with register_plugin_dir
    # model_collection_path = os.path.join(__file__, 'models') 

    def __init__(self):
        ''' Initializing mount, or registering a plugin?    '''
        self.model_collection_path = ''
        self.models = {}
        # if not hasattr(self, 'models'):
        #     print('Mount initialized')
        #     # self.models = ModelStruct(Registrar)
        #     self.models = {}
        # else:
        #     print('Model registered')
        #     self.register_model(self)
    
    def register_model(self, model):
        ''' Registration logic + append to plugins struct '''
        print(model)
        # model = model(model.__class__.__name__) #< Init the plugin
        model = model()
        # self.models[model.__class__.__name__] = model
        # self.register_model_dir(model.model_dir)
        wrapper = model.wrapper
        module_name = model.module_name
        ## Add an if-else to check if model sent is 'default' or other versions
        self.models[(module_name, model.__class__.__name__)] = {'default':[model, wrapper]}
        # List of tuples
        # dict of dict with keys as version name for models
        # {
        # default:[(model, wrapper)],
        # V1:[(model, wrapper)],
        # V2:[(model, wrapper)]
        # }
    
    # @staticmethod
    def register_model_dir(self, model_collection_path):
        ''' This function sets the plugin path to be searched   '''
        if os.path.isdir(model_collection_path):
            # Registrar.model_collection_path = model_collection_path
            self.model_collection_path = model_collection_path
        else:
            raise EnvironmentError('%s is not a directory' % model_collection_path)
        
    # @staticmethod
    def find_models(self): # Change name to `load...`
        ''' Traverse registered plugin directory and import non-loaded modules  '''
        # model_collection_path = Registrar.model_collection_path
        # model_collection_path = '/home/shakkeel/Desktop/plugin_tests/selfregistering_approach/imly/arch'
        if not os.path.isdir(self.model_collection_path):
            raise EnvironmentError('%s is not a directory' % self.model_collection_path) # Make the error more informative
        
        for file_ in os.listdir(self.model_collection_path):
            if file_.endswith('.py') and file_ != '__init__.py':
                module = file_[:-3] #< Strip extension
                mod_obj = globals().get(module)
                if mod_obj is None:
                    f, filename, desc = imp.find_module(
                        module, [self.model_collection_path])
                    print(module)
                    globals()[module] = mod_obj = imp.load_module(
                        module, f, filename, desc)


    def __getitem__(self, key, retry=True):
        ''' Re-implement __getitem__ to scan for plugins if key missing  '''
        try:
            return self.models[key]
        except KeyError:
            # if default != False:
            #     return default
            if retry:
                self.find_models()
                return self.__getitem__(key, False)
            else:
                raise KeyError(
                    'Model "%s" not found in model_dir "%s"' % (
                        key, self.mount.model_collection_path
                    )
                )
                




# class ModelStruct(dict): ## Move to Registrar
#     '''
#         Subclass dict, re-implement __getitem__ to scan for plugins
#         if a requested key is missing
#     '''
#     def __init__(self, cls, *args, **kwargs):
#         '''
#             Init, set mount to PlugPyMount master instance
#             @param  PlugPyMount cls
#         '''
#         self.mount = cls
#         super(ModelStruct, self).__init__(*args,**kwargs)

#     def __getitem__(self, key, retry=True, default=False):
#         ''' Re-implement __getitem__ to scan for plugins if key missing  '''
#         try:
#             return super(ModelStruct, self).__getitem__(key)
#         except KeyError:
#             # if default != False:
#             #     return default
#             if retry:
#                 self.mount.find_models()
#                 return self.__getitem__(key, False)
#             else:
#                 raise KeyError(
#                     'Model "%s" not found in plugin_dir "%s"' % (
#                         key, self.mount.model_collection_path
#                     )
#                 )

registrar = Registrar()

# class registry(metaclass=Registrar): ## Change to modellib.model ## model_registry
#     pass
#     # def __init__(self, model_class):
#     #   self.model_class = model_class