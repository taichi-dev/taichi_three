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
        self.banner = f"\n{'*' * 43}\n**      Taichi Programming Language      **\n{'*' * 43}"
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
               f'Docs:   https://taichi.rtfd.io/{uri}\n' \
               f'GitHub: https://github.com/taichi-dev/taichi\n' \
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
    def video(self, arguments: list = sys.argv[2:]):
        """Make a video using *.png files in the current directory"""
        parser = argparse.ArgumentParser(prog='t3 video',
                                         description=f"{self.video.__doc__}")
        parser.add_argument("inputs", nargs='*', help="PNG file(s) as inputs")
        parser.add_argument('-o',
                            '--output',
                            required=False,
                            default=Path('./video.mp4').resolve(),
                            dest='output_file',
                            type=lambda x: Path(x).resolve(),
                            help="Path to output MP4 video file")
        parser.add_argument('-f',
                            '--framerate',
                            required=False,
                            default=24,
                            dest='framerate',
                            type=int,
                            help="Frame rate of the output MP4 video")
        args = parser.parse_args(arguments)

    @register
    def run(self, arguments: list = sys.argv[2:]):
        """Run a single script"""
        parser = argparse.ArgumentParser(prog='t3 run',
                                         description=f"{self.run.__doc__}")
        parser.add_argument(
            'filename',
            help='A single (Python) script to run with Taichi, e.g. render.py')
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
        return runpy.run_path('setup.py')

    @register
    def repl(self, arguments: list = sys.argv[2:]):
        """Start Taichi REPL / Python shell with 'import taichi as ti'"""
        parser = argparse.ArgumentParser(prog='ti repl',
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
        parser = argparse.ArgumentParser(prog='ti lint',
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
