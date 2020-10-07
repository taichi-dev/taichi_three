# Miscellaneous


Here's a collection of some random yet useful stuffs you might need.


## Install from latest master branch

If you'd like to keep catching up with latest Taichi THREE updates, please clone it from [our GitHub repository](https://github.com/taichi-dev/taichi_three). Then build and install it:

```bash
git clone https://github.com/taichi-dev/taichi_three.git
# try this mirror repository on Gitee if the above is too slow:
# git clone https://gitee.com/archibate/taichi_three.git

cd taichi_three
python3 -m pip install -r requirements.txt  # install `taichi` and `taichi-glsl`
python3 -m pip install wheel                # required for the next step
python3 setup.py bdist_wheel                # create a `.whl` file
pip install -U dist/taichi_three-0.0.6-py3-none-any.whl
```


## Introducing Taichi

[Taichi](https://github.com/taichi-dev/taichi) is a programming langurage that is highly-embed into Python that basically allows you to write high-performance GPU programs in Python syntax.
And Taichi THREE is a soft rendering pipeline based on Taichi.
You may find the documentation of Taichi [here](https://taichi.rtfd.io).


## Command line interface

Installing ``taichi_three`` successfully should also allows you to invoke it from command line. 
E.g., to visualize a OBJ model located at `/path/to/model.obj`:

```py
python3 -m taichi_three show /path/to/model.obj
```

## Mirror sites

Main site: https://t3.142857.red/
GitHub pages: https://taichi-dev.github.io/taichi_three/
Gitee pages: https://archibate.gitee.io/taichi_three/