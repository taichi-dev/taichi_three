import setuptools
import tina

setuptools.setup(
    name='taichi-tina',
    version='.'.join(map(str, tina.__version__)),
    author=tina.__author__.split('<')[0][:-1],
    author_email=tina.__author__.split('<')[1][:-1],
    url='https://github.com/taichi-dev/taichi_three',
    description='A real-time soft renderer based on the Taichi programming language',
    long_description=tina.__doc__,
    license=tina.__license__,
    keywords=['graphics', 'renderer'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Games/Entertainment :: Simulation',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'taichi',
        'transformations'
    ],
    packages=setuptools.find_packages(),
)
