@echo off
set workingDir=%~dp0
set pywinDir=%workingDir%pywin37

echo "creating conda... into %pywinDir%"
set condaRoot=D:\Anaconda\Anaconda38
REM call %condaRoot%\Scripts\activate.bat
REM conda config --remove-key channels
REM call conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
REM call conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
REM call conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
REM call conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
REM call conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
REM call conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
REM call conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/
REM call conda config --add channels http://mirrors.aliyun.com/pypi/simple/
REM call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
REM call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
REM call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
call %condaRoot%\Scripts\conda.exe create --yes -p %pywinDir% python=3.7
REM call %pywinDir%\Scripts\pip.exe install pyqt5==5.15.2 pyqt5-tools==5.15.2.3.2 opencv-python==3.4.2.17 imageio==2.8.0 onnxruntime==1.7.0 pyinstaller==4.10


