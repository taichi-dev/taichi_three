@eval('lambda x: x()')
def mock():
    # An artwork presented by github.com/archibate

    import importlib
    import threading
    import builtins
    import os

    lines_cache = {}
    global_lock = threading.Lock()
    mod_attrs_cache = {}

    def mock(globals):
        """
        Set up lazy import in a module. To use:

        # mypkg/__init__.py
        if __import__('lazyimp').mock():
            from .foo import *

        # mypkg/foo.py
        def hello():
            print('hello')

        Then calling `mypkg.hello()` will import `mypkg.foo` just-in-time.
        """
        if hasattr(mock, 'disable'):
            return True

        def make_getattr(this_file, this_module):
            def wrapped(name):
                with global_lock:
                    def get_module_attrs(path):
                        if path not in lines_cache:
                            with open(path, 'r', encoding='utf-8') as f:
                                lines_cache[path] = f.readlines()
                        lines = lines_cache[path]
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
                                module = importlib.import_module('.' + mod_name, packed)
                                yield getattr(module, name)

                    directory = os.path.dirname(os.path.abspath(this_file))
                    for attr in search_module(directory, this_module):
                        globals[name] = attr
                        return attr

                    raise AttributeError(name)

            return wrapped

        globals['__getattr__'] = make_getattr(globals['__file__'], globals['__package__'])
        return False

    return mock