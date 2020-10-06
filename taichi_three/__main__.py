import argparse
import math
import os
import random
import runpy
import shutil
import sys
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path


def timer(func):
    """Function decorator to benchmark a function runnign time."""
    import timeit

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - start
        print(f">>> Running time: {elapsed:.2f}s")
        return result

    return wrapper


def registerableCLI(cls):
    """Class decorator to register methodss with @register into a set."""
    cls.registered_commands = set([])
    for name in dir(cls):
        method = getattr(cls, name)
        if hasattr(method, 'registered'):
            cls.registered_commands.add(name)
    return cls


def register(func):
    """Method decorator to register CLI commands."""
    func.registered = True
    return func


@registerableCLI
class Main:
    def __init__(self, test_mode: bool = False):
        self.banner = f"\n{'*' * 37}\n**      Taichi THREE Renderer      **\n{'*' * 37}"
        print(self.banner)

        print(self._get_friend_links())

        parser = argparse.ArgumentParser(description="Taichi CLI",
                                         usage=self._usage())
        parser.add_argument('command',
                            help="command from the above list to run")

        # Flag for unit testing
        self.test_mode = test_mode

        self.main_parser = parser

    @timer
    def __call__(self):
        # Print help if no command provided
        if len(sys.argv[1:2]) == 0:
            self.main_parser.print_help()
            return 1

        # Parse the command
        args = self.main_parser.parse_args(sys.argv[1:2])

        if args.command not in self.registered_commands:
            # TODO: do we really need this?
            if args.command.endswith(".py"):
                Main._exec_python_file(args.command)
            else:
                print(f"{args.command} is not a valid command!")
                self.main_parser.print_help()
            return 1

        return getattr(self, args.command)(sys.argv[2:])

    def _get_friend_links(self):
        uri = 'en/stable'
        try:
            import locale
            if 'zh' in locale.getdefaultlocale()[0]:
                uri = 'zh_CN/latest'
        except:
            pass
        return '\n' \
               f'GitHub: https://github.com/taichi-dev/taichi_three\n' \
               f'Forum:  https://forum.taichi.graphics\n'

    def _usage(self) -> str:
        """Compose deterministic usage message based on registered_commands."""
        # TODO: add some color to commands
        msg = "\n"
        space = 20
        for command in sorted(self.registered_commands):
            msg += f"    {command}{' ' * (space - len(command))}|-> {getattr(self, command).__doc__}\n"
        return msg

    @register
    def conv(self, arguments: list = sys.argv[2:]):
        """Conversation between OBJ and NPZ file formats"""
        parser = argparse.ArgumentParser(prog='t3 conv',
                                         description=f"{self.conv.__doc__}")
        parser.add_argument(
            'input',
            help='File name of the OBJ/NPZ model as input, e.g. monkey.obj')
        parser.add_argument(
            'output',
            help='File name of the NPZ/OBJ model as output, e.g. monkey.npz')
        args = parser.parse_args(arguments)

        import taichi_three as t3
        obj = t3.readobj(args.input)
        t3.writeobj(args.output, obj)

    @register
    def info(self, arguments: list = sys.argv[2:]):
        """Display informations of an OBJ/NPZ model"""
        parser = argparse.ArgumentParser(prog='t3 show',
                                         description=f"{self.show.__doc__}")
        parser.add_argument(
            'filename',
            help='File name of the OBJ/NPZ model to visualize, e.g. monkey.obj')
        args = parser.parse_args(arguments)

        import taichi_three as t3
        obj = t3.readobj(args.filename)
        print('vertices:', len(obj['vp']))
        print('normals:', len(obj['vn']))
        print('texcoors:', len(obj['vt']))
        print('maxpoly:', max(len(f) for f in obj['f']))
        print('faces:', len(obj['f']))

    @register
    def show(self, arguments: list = sys.argv[2:]):
        """Visualize an OBJ/NPZ model using Taichi THREE"""
        parser = argparse.ArgumentParser(prog='t3 show',
                                         description=f"{self.show.__doc__}")
        parser.add_argument(
            'filename',
            help='File name of the OBJ/NPZ model to visualize, e.g. monkey.obj')
        parser.add_argument('-s', '--scale', default=0.75,
                type=float, help='Specify a scale parameter')
        parser.add_argument('-o', '--ortho',
                action='store_true', help='Display in orthogonal mode')
        parser.add_argument('-l', '--lowp',
                action='store_true', help='Shade faces by interpolation')
        parser.add_argument('-t', '--texture',
                type=str, help='Path to texture to bind')
        parser.add_argument('-n', '--normtex',
                type=str, help='Path to normal map to bind')
        parser.add_argument('-m', '--metallic',
                type=str, help='Path to metallic map to bind')
        parser.add_argument('-r', '--roughness',
                type=str, help='Path to roughness map to bind')
        parser.add_argument('-a', '--arch', default='cpu',
                type=str, help='Backend to use for rendering')
        args = parser.parse_args(arguments)

        import taichi as ti
        import taichi_three as t3
        import numpy as np

        ti.init(getattr(ti, args.arch))

        scene = t3.Scene()
        obj = t3.readobj(args.filename, scale=args.scale)
        model = (t3.Model if args.lowp else t3.ModelPP).from_obj(obj)
        if args.texture is not None:
            model.add_texture('color', ti.imread(args.texture))
        if args.normtex is not None:
            model.add_texture('normal', ti.imread(args.normtex))
        if args.metallic is not None:
            model.add_texture('metallic', ti.imread(args.metallic))
        if args.roughness is not None:
            model.add_texture('roughness', ti.imread(args.roughness))
        scene.add_model(model)
        camera = t3.Camera()
        if args.ortho:
            camera.type = camera.ORTHO
        scene.add_camera(camera)
        light = t3.Light([0.4, -1.5, 0.8])
        scene.add_light(light)

        gui = ti.GUI('Model', camera.res)
        while gui.running:
            gui.get_event(None)
            gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
            camera.from_mouse(gui)
            scene.render()
            gui.set_image(camera.img)
            gui.show()

    @register
    def run(self, arguments: list = sys.argv[2:]):
        """Run a single Python script"""
        parser = argparse.ArgumentParser(prog='t3 run',
                                         description=f"{self.run.__doc__}")
        parser.add_argument(
            'filename',
            help='A single (Python) script to run with Taichi THREE, e.g. render.py')
        args = parser.parse_args(arguments)

        runpy.run_path(args.filename)

    @register
    def dist(self, arguments: list = sys.argv[2:]):
        """Build package and release"""
        parser = argparse.ArgumentParser(prog='t3 dist',
                                         description=f"{self.dist.__doc__}")
        args = parser.parse_args(arguments)

        os.chdir(os.path.join(os.path.dirname(__file__), '..'))
        sys.argv = ['setup.py', 'bdist_wheel']
        runpy.run_path('setup.py')

    @register
    def repl(self, arguments: list = sys.argv[2:]):
        """Start Taichi REPL / Python shell with 'import taichi as ti'"""
        parser = argparse.ArgumentParser(prog='t3 repl',
                                         description=f"{self.repl.__doc__}")
        args = parser.parse_args(arguments)

        def local_scope():
            import taichi as ti
            import taichi_glsl as tl
            import taichi_three as t3
            import numpy as np
            import math
            import time
            try:
                import IPython
                IPython.embed()
            except ImportError:
                import code
                __name__ = '__console__'
                code.interact(local=locals())

        local_scope()

    @register
    def lint(self, arguments: list = sys.argv[2:]):
        """Run pylint checker for the Python codebase of Taichi"""
        parser = argparse.ArgumentParser(prog='t3 lint',
                                         description=f"{self.lint.__doc__}")
        # TODO: support arguments for lint specific files
        args = parser.parse_args(arguments)

        options = [os.path.dirname(__file__)]

        from multiprocessing import cpu_count
        threads = min(8, cpu_count())
        options += ['-j', str(threads)]

        # http://pylint.pycqa.org/en/latest/user_guide/run.html
        # TODO: support redirect output to lint.log
        import pylint.lint
        pylint.lint.Run(options)


def main():
    cli = Main()
    return cli()


if __name__ == "__main__":
    sys.exit(main())
