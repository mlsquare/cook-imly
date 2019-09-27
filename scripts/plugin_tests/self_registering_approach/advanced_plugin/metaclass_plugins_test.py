import os
import imp

class PlugPyStruct(dict):
    '''
        Subclass dict, reimplement __getitem__ to scan for plugins
        if a requested key is missing
    '''
    def __init__(self, cls, *args, **kwargs):
        '''
            Init, set mount to PlugPyMount master instance
            @param  PlugPyMount cls
        '''
        # print('from struct--', cls)
        self.mount = cls
        # self.iter_refresh = False
        super(PlugPyStruct, self).__init__(*args,**kwargs)

    def __getitem__(self, key, retry=True, default=False):
        ''' Reimplement __getitem__ to scan for plugins if key missing  '''
        try:
            print('__getitem__ key -- ', key)
            return super(PlugPyStruct, self).__getitem__(key)
        except KeyError:
            if default != False:
                return default ## What is this 'default' for?
            elif retry:
                self.mount.find_plugins()
                return self.__getitem__(key, False)
            else:
                raise KeyError(
                    'Plugin "%s" not found in plugin_dir "%s"' % (
                        key, self.mount.plugin_path
                    )
                )
    
    # def set_iter_refresh(self, refresh=True):
    #     '''
    #         Toggle flag to search for new plugins before iteration
    #         @param  bool    refresh     Whether to refresh before iteration
    #     '''
    #     self.iter_refresh = refresh
    
    # def __iter__(self):
    #     ''' Reimplement __iter__ to allow for optional plugin refresh   '''
    #     if self.iter_refresh:   self.mount.find_plugins()
    #     return super(PlugPyStruct, self).__iter__()
    
    # def values(self):
    #     ''' Reimplement values to allow for optional plugin refresh   '''
    #     if self.iter_refresh:   self.mount.find_plugins()
    #     return super(PlugPyStruct, self).values()
    
    # def keys(self):
    #     ''' Reimplement keys to allow for optional plugin refresh   '''
    #     if self.iter_refresh:   self.mount.find_plugins()
    #     return super(PlugPyStruct, self).keys()
    
    # def items(self):
    #     ''' Reimplement items to allow for optional plugin refresh   '''
    #     if self.iter_refresh:   self.mount.find_plugins()
    #     return super(PlugPyStruct, self).items()
    
    # def itervalues(self):
    #     ''' Reimplement itervalues to allow for optional plugin refresh   '''
    #     if self.iter_refresh:   self.mount.find_plugins()
    #     return super(PlugPyStruct, self).itervalues()
    
    # def iterkeys(self):
    #     ''' Reimplement iterkeys to allow for optional plugin refresh   '''
    #     if self.iter_refresh:   self.mount.find_plugins()
    #     return super(PlugPyStruct, self).iterkeys()
    
    # def iteritems(self):
    #     ''' Reimplement iteritems to allow for optional plugin refresh   '''
    #     if self.iter_refresh:   self.mount.find_plugins()
    #     return super(PlugPyStruct, self).iteritems()


class PlugPyMount(type):
    ''' This class acts as a mount point for our plugins    '''
    
    # Defaults to ./plugins, change with register_plugin_dir
    plugin_path = os.path.join(__file__, 'plugins')
    
    def __init__(self, name, bases, attr):
        ''' Initializing mount, or registering a plugin?    '''
        ## Why does Metaclass init get triggered when the class-object
        ## is initialized?
        if not hasattr(self, 'plugins'):
            print('Mount initialization')
            ## Why do we have to send PlugPyMount as a parent?
            self.plugins = PlugPyStruct(PlugPyMount)
            ## Why do we need this if the returned value is an empty dict?
            ## Why is an empty dict returned?
            ## How does PLugPyStruct extend search functionality?
        else:
            print('Registering plugin -- ', type(self))
            self.register_plugin(self)
    
    def register_plugin(self, plugin):
        ''' Registration logic + append to plugins struct '''
        plugin = plugin() #< Init the plugin
        print('From register_plugin -- ', type(plugin))
        self.plugins[plugin.__class__.__name__] = plugin
    
    @staticmethod
    def register_plugin_dir(plugin_path):
        ''' This function sets the plugin path to be searched   '''
        if os.path.isdir(plugin_path):
            PlugPyMount.plugin_path = plugin_path
        else:
            raise EnvironmentError('%s is not a directory' % plugin_path)
        
    @staticmethod
    def find_plugins():
        ''' Traverse registered plugin directory and import non-loaded modules  '''
        plugin_path = PlugPyMount.plugin_path
        if not os.path.isdir(plugin_path):
            raise EnvironmentError('%s is not a directory' % plugin_path)
        
        ## Cross check what's causing error here
        print('from find_plugin -- ',plugin_path)
        for file_ in os.listdir(plugin_path):
            if file_.endswith('.py') and file_ != '__init__.py':
                module = file_[:-3] #< Strip extension
                mod_obj = globals().get(module)
                if mod_obj is None:
                    f, filename, desc = imp.find_module(
                        module, [plugin_path])
                    globals()[module] = mod_obj = imp.load_module(
                        module, f, filename, desc)


class PlugPy(metaclass=PlugPyMount):
        ''' Default PlugPy implementation, metaclasses PlugPyMount '''

# class TestPlugin(PlugPy):
#     initialized = False
#     def __init__(self):
#         print('Initializing TestPlugin')
#         self.initialized = True
#     def run(self):
#         print('Running TestPlugin')
#         return True



# class Test2(PlugPy):
#     initialized = False
#     def __init__(self):
#         print('Initializing TestPlugin')
#         self.initialized = True
#     def run(self):
#         print('Running TestPlugin')
#         return True


# if __name__ == '__main__':
#     # PlugPy.register_plugin_dir('/home/shakkeel/Desktop/plugin_tests')
#     print(PlugPy.plugins)
#     PlugPy.plugins['TestPlugin'].run()
#     # PlugPy.find_plugins()