from metaclass_plugins_test import PlugPy
import plugpytest

class Test3(PlugPy):
    initialized = False
    def __init__(self):
        print('Initializing Test3')
        self.initialized = True
    def run(self):
        print('Running Test3')
        return True

class Test4(PlugPy):
    initialized = False
    def __init__(self):
        print('Initializing Test4')
        self.initialized = True
    def run(self):
        print('Running Test4')
        return True