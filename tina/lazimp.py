@eval('lambda x: x()')
def lazyguard():
    # An artwork presented by github.com/archibate

    import importlib
    import threading
    import builtins
    import inspect
    import os

    search_lock = threading.Lock()
    mod_attrs_cache = {}

    @eval('lambda x: x()')
    class lazyguard:
        """
        Set up lazy import in a module. To use:

        # mypkg/__init__.py
        if __import__('lazyimp').lazyguard:
            from .foo import *

        # mypkg/foo.py
        def hello():
            print('hello')

        Then calling `mypkg.hello()` will import `mypkg.foo` just-in-time.
        """

        def __bool__(self):
            def make_getattr(this_file, this_module):
                def wrapped(name):
                    def get_module_attrs(path):
                        with open(path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        magic = "'''''"
                        has_magic = False
                        had_magic = False
                        mod_attrs = []
                        for line in lines:
                            if has_magic:
                                if line.startswith(magic):
                                    has_magic = False
                                    continue
                                for x in line.split():
                                    mod_attrs.append(x.strip())
                            elif line.startswith(magic):
                                has_magic = True
                                had_magic = True
                        if not had_magic:
                            for line in lines:
                                for magic in ['class ', 'def ']:
                                    if line.startswith(magic):
                                        i = len(magic)
                                        while i < len(line) and (line[i].isalnum() or line[i] == '_'):
                                            i += 1
                                        attr = line[len(magic):i].strip()
                                        if len(attr) and not attr.startswith('_'):
                                            mod_attrs.append(attr)
                        return frozenset(mod_attrs)

                    def search_module(directory, packed):
                        for file in os.listdir(path=directory):
                            path = os.path.join(directory, file)
                            if os.path.isdir(path) and os.path.isfile(os.path.join(path, '__init__.py')):
                                yield from search_module(path, packed + '.' + file)
                                continue
                            if not os.path.isfile(path) or not file.endswith('.py'):
                                continue
                            if path not in mod_attrs_cache:
                                mod_attrs_cache[path] = get_module_attrs(path)
                            mod_attrs = mod_attrs_cache[path]
                            mod_name = file[:-3]
                            if name in mod_attrs:
                                def getter():
                                    module = importlib.import_module('.' + mod_name, packed)
                                    return getattr(module, name)
                                yield getter

                    directory = os.path.dirname(os.path.abspath(this_file))
                    search = iter(search_module(directory, this_module))
                    try:
                        with search_lock:
                            getter = next(search)
                        globals[name] = getter()
                    except StopIteration:
                        raise AttributeError("Module '" + this_module + "' has no attribute named '" + name + "'") from None

                    return globals[name]

                return wrapped

            globals = inspect.stack()[1][0].f_globals
            globals['__getattr__'] = make_getattr(globals['__file__'], globals['__package__'])
            return False

    return lazyguard
