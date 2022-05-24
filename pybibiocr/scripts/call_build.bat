REM off
set exename=bibiocr
set currDir=%~dp0
set workingDir=%currDir%..\..
set pywinDir=%workingDir%\pywin37
set pysrcDir=%workingDir%\pybibiocr



set PATH=%pysrcDir%\;%pywinDir%\; %pywinDir%\Scripts;%pywinDir%\Library\bin;%PATH%

call cd %pysrcDir%

REM call %pywinDir%\Scripts\pip.exe install Cython
REM call %pywinDir%\Scripts\pip.exe install pyinstaller

call Xcopy /S /I /E /Y %pysrcDir%\binary\geos_c.dll %pywinDir%\Library\bin

REM call %pywinDir%\python.exe %pysrcDir%\setup.py bdist build_ext --inplace
call %pywinDir%\python.exe %pysrcDir%\setup.py build_ext --inplace
REM call del %pysrcDir%\bibiocr.*.pyd


call %pywinDir%\Scripts\pyinstaller.exe --clean -y %pysrcDir%\bibiocr.spec
call Xcopy /S /I /E /Y %pywinDir%\Lib\site-packages\Shapely.libs %pysrcDir%\dist\%exename%\Shapely.libs
call Xcopy /S /I /E /Y %pysrcDir%\build\lib.win-amd64-3.7\pybibiocr\mainwindow.pyd %pysrcDir%\dist\%exename%\
call Xcopy /S /I /E /Y %pysrcDir%\build\lib.win-amd64-3.7\pybibiocr\onnx %pysrcDir%\dist\%exename%\onnx
call Xcopy /S /I /E /Y %pysrcDir%\binary\snip.exe %pysrcDir%\dist\%exename%\



REM ====================================================================================================================
REM call Xcopy /S /I /E /Y %pywinDir%\Lib\site-packages\paddle %pysrcDir%\dist\%exename%\paddle
REM call Xcopy /S /I /E /Y %pywinDir%\Lib\site-packages\paddleocr %pysrcDir%\dist\%exename%\paddleocr

REM call %pywinDir%\Scripts\pyinstaller.exe --clean --exclude matplotlib -i %pysrcDir%\qtocr.ico --noconfirm --name %exename% qtocr.py --hidden-import skimage --hidden-import framework_pb2 --hidden-import paddlepaddle --hidden-import paddle --hidden-import paddleocr --hidden-import pyqt5 --hidden-import pyqt5-tools --hidden-import mainwindow --add-data "%pysrcDir%\ppocr;%pysrcDir%\dist\%exename%\ppocr"

REM call mkdir %pysrcDir%\dist\ocrtool\images
REM call %pywinDir%\Scripts\pip.exe install -r %pysrcDir%\requirements11.txt

REM call %pywinDir%\python.exe setup.py install
REM start %SystemRoot%\explorer.exe "%pywinDir%\Lib\site-packages\paddle\dataset\"
REM mshta vbscript:msgbox("Please Change %pywinDir%\Lib\site-packages\paddle\dataset\image.py",64,"Remove Threads Setting")(window.close)


