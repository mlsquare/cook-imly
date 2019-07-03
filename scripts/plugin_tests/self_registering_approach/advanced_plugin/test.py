import os, imp

# plugin_path = '/home/shakkeel/Desktop/plugin_tests/selfregistering_approach/advanced_plugin/plugins'
plugin_path = '/home/shakkeel/Desktop/plugin_tests'

for file_ in os.listdir(plugin_path):
    if file_.endswith('.py') and file_ != '__init__.py':
        module = file_[:-3] #< Strip extension
        mod_obj = globals().get(module)
        if mod_obj is None:
            f, filename, desc = imp.find_module(
                module, [plugin_path])
            globals()[module] = mod_obj = imp.load_module(
                module, f, filename, desc)



# # # # # # # # # # # # # # # # # # # # 

# from metaclass_plugins_test import PlugPyStruct, PlugPyMount, PlugPy
# # import plugpytest

# # test_instance = PlugPyStruct(PlugPyMount)

# # print('from test.py -- ',type(test_instance))


# class TestPlugin(PlugPy):
#     initialized = False
#     def __init__(self):
#         print('Initializing TestPlugin')
#         self.initialized = True
#     def run(self):
#         print('Running TestPlugin')
#         return True

# test_class = PlugPy.plugins['TestPlugi']



# print(PlugPy.plugins)

# print(test_class)

# test_class.run()


## Notes
# 1) PlugPyStruct to override typical dict behaviour.
# A normal self.plugins = {} would not be sufficient. Why? -- extending with find_plugin() method.
# Is there a better way to do this?
# 2) 'PlugPyMount' is sent as an argument. This is to access find_plugin() from the mount.

## IMLY use cases
# 1) Registry - To ease the mapping process.
# 2) Metaclasses - To ensure that important methods are not left undeclared.
# 3) Support models as plugins(can be imported from cloud)