import taichi_three as t3
import setuptools

setuptools.setup(
    name=t3.__package__,
    version='.'.join(map(str, t3.__version__)),
    author=t3.__author__.split('<')[0][:-1],
    author_email=t3.__author__.split('<')[1][:-1],
    url='https://github.com/taichi-dev/taichi_three',
    description='A Taichi extension library to render 3D scenes',
    long_description=t3.__doc__,
    license='MIT',
    keywords=['graphics', 'renderer'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Games/Entertainment :: Simulation',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requies=[
        'taichi>=0.6.38',
        'taichi-glsl',
    ],
    packages=setuptools.find_packages(),
)
