@eval('lambda x: x()')
def lazyguard():
    # Lazy import - an artwork presented by github.com/archibate

    import importlib
    import threading
    import builtins
    import inspect
    import os

    def reload_package(package):
        import os
        import types
        import importlib

        assert(hasattr(package, "__package__"))
        fn = package.__file__
        fn_dir = os.path.dirname(fn) + os.sep
        module_visit = {fn}
        del fn

        def reload_recursive_ex(module):
            importlib.reload(module)
            for module_child in dict(vars(module)).values():
                if isinstance(module_child, types.ModuleType):
                    fn_child = getattr(module_child, "__file__", None)
                    if (fn_child is not None) and fn_child.startswith(fn_dir):
                        if fn_child not in module_visit:
                            # print("reloading:", fn_child, "from", module)
                            module_visit.add(fn_child)
                            reload_recursive_ex(module_child)

        reload_recursive_ex(package)

    search_lock = threading.Lock()
    mod_attrs_cache = {}

    class Lazyguard:
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

        def get_mod_attrs_cache(self):
            return mod_attrs_cache

        def __bool__(self):
            globals = inspect.stack()[1][0].f_globals
            return self.make(globals, hook_getattr=True)

        def __call__(self, hook_getattr=False):
            globals = inspect.stack()[1][0].f_globals
            return self.make(globals, hook_getattr=False)

        def make(self, globals, hook_getattr):
            def make(this_file, this_module):
                assert this_module is not None

                def getattr_cb(name):
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
                        written_lazy_names.add(name)
                    except StopIteration:
                        raise AttributeError("Module '" + this_module + "' has no attribute named '" + name + "'") from None

                    return globals[name]

                def reload_cb():
                    with search_lock:
                        for name in written_lazy_names:
                            # print('del', name)
                            del globals[name]
                        written_lazy_names.clear()
                        mod_attrs_cache.clear()
                    module = importlib.import_module('.', this_module)
                    reload_package(module)

                return getattr_cb, reload_cb

            written_lazy_names = set()
            getattr_cb, reload_cb = make(globals['__file__'], globals['__package__'])
            if hook_getattr:
                globals['__getattr__'] = getattr_cb
            globals['__lazyreload__'] = reload_cb
            return False

    class DisableLazyguard:
        def __bool__(self):
            return True

    return Lazyguard()
