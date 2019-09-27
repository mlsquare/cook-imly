registry = []

class Plugin(type):
    def __init__(cls, name, bases, dict):
        # type.__init__(name, bases, dict)
        registry.append((name, cls))


## Plugins
class testPlugin1(metaclass=Plugin):
    pass


class testPlugin2(metaclass=Plugin):
    pass


# loop over registered plugins
for name, cls in registry:
    if cls is not Plugin:
    	print(name, cls)

print(registry)