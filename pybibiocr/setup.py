# from distutils.core import setup
from setuptools import setup, find_packages
from Cython.Build import cythonize

pyfiles = ["bibiocr.py",
           "mainwindow.py"]
setup(
    name="bibiocr",
    version="1.0",
    packages=find_packages(),
    py_modules=[  # 在 package 之外添加两个独立的 module
        'onnx'
    ],
    ext_modules=cythonize(pyfiles),

    include_package_data=True,
    install_requires=[
        "pyqt5==5.15.2",
        "pyqt5-tools==5.15.2.3.2",
        "onnxruntime==1.7.0",
        "imageio==2.8.",
        "opencv-python==3.4.2.17",
        "pillow==9.1.0",
        "Shapely==1.7.0",
        "pyclipper==1.2.0"
    ],
    scripts=pyfiles,
    package_data={
        'bibiocr': ['config.json', 'bibiocr.ico', 'onnx/models/*']
    }
)
