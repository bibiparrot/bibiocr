# from distutils.core import setup
from setuptools import setup, find_packages
from Cython.Build import cythonize

# import compileall
# compileall.compile_dir('onnx', force=True)

pyfiles = ["bibiocr.py",
           "mainwindow.py",
           "./onnx/cnocr_lite/angnet/angle.py",
           # "./onnx/cnocr_lite/angnet/__init__.py",
           "./onnx/cnocr_lite/config.py",
           "./onnx/cnocr_lite/crnn/CRNN.py",
           "./onnx/cnocr_lite/crnn/keys.py",
           "./onnx/cnocr_lite/crnn/util.py",
           # "./onnx/cnocr_lite/crnn/__init__.py",
           "./onnx/cnocr_lite/dbnet/dbnet_infer.py",
           "./onnx/cnocr_lite/dbnet/decode.py",
           "./onnx/cnocr_lite/model.py",
           "./onnx/cnocr_lite/ocr_run.py",
           "./onnx/cnocr_lite/utils.py",
           # "./onnx/cnocr_lite/__init__.py",
           "./onnx/rapid_ocr/ch_ppocr_mobile_v2_cls/text_cls.py",
           "./onnx/rapid_ocr/ch_ppocr_mobile_v2_cls/utils.py",
           # "./onnx/rapid_ocr/ch_ppocr_mobile_v2_cls/__init__.py",
           "./onnx/rapid_ocr/ch_ppocr_mobile_v2_det/text_detect.py",
           "./onnx/rapid_ocr/ch_ppocr_mobile_v2_det/utils.py",
           # "./onnx/rapid_ocr/ch_ppocr_mobile_v2_det/__init__.py",
           "./onnx/rapid_ocr/ch_ppocr_mobile_v2_rec/text_recognize.py",
           "./onnx/rapid_ocr/ch_ppocr_mobile_v2_rec/utils.py",
           # "./onnx/rapid_ocr/ch_ppocr_mobile_v2_rec/__init__.py",
           # "./onnx/rapid_ocr/__init__.py",
           # "./onnx/__init__.py",
           ]
setup(
    name="pybibiocr",
    version="1.0",
    packages=find_packages(),
    py_modules=[  # 在 package 之外添加独立的 module
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
        'pybibiocr': ['config.json', 'bibiocr.ico', 'onnx/models/*']
    }
)
