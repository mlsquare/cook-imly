from metaclass_plugins_test import PlugPy

class Test2(PlugPy):
    initialized = False
    def __init__(self):
        print('Initializing Test2')
        self.initialized = True
    def run(self):
        print('Running Test2')
        return True

PlugPy.register_plugin_dir('/home/shakkeel/Desktop/plugin_tests/selfregistering_approach/advanced_plugin/plugins')
PlugPy.plugins['Test4'].run()
# PlugPy.find_plugins()