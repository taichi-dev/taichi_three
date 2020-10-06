# Installation

First of all, let's install Taichi THREE. Make sure you're using Python **3.6/3.7/3.8**, and **64-bit**. And then install Taichi THREE via `pip`:

```bash
python3 -m pip install taichi_three
```

To verify the installation, type this command into the Python shell:

```py
import taichi_three as t3
```

## Command line interface

Installing ``taichi_three`` successfully should also allows you to invoke it from command line. 
E.g., to visualize a OBJ model:

```py
python3 -m taichi_three show /path/to/model.obj
```

## Introducing Taichi

[Taichi](https://github.com/taichi-dev/taichi) is a programming langurage that is highly-embed into Python that basically allows you to write high-performance GPU programs in Python syntax.
And Taichi THREE is a soft rendering pipeline based on Taichi.
You may find the documentation of Taichi [here](https://taichi.rtfd.io).

## Troubleshooting

See https://taichi.readthedocs.io/en/latest/install.html#troubleshooting for issues related to Taichi.

If you encounter problems using Taichi THREE, please let me know by [opening an issue at GitHub](https://github.com/taichi-dev/taichi_three/issues/new), many thanks!