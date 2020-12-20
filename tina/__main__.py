def main(*args):
    cmd, *args = args
    if cmd == 'mesh':
        from .cli.mesh import main
        return main(*args)
    if cmd == 'volume':
        from .cli.volume import main
        return main(*args)
    if cmd == 'particles':
        from .cli.particles import main
        return main(*args)
    else:
        print('bad command:', cmd)
        exit(1)


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
