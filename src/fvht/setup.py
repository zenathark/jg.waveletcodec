from distutils.core import setup, Extension
from Cython.Distutils import build_ext


if __name__ == "__main__":
    setup(
        cmdclass={'build_ext', build_ext},
        ext_modules=[Extension("fvht", ["_fvht.pyx"])]
    )
