
@echo off
set currDir=%~dp0
set workingDir=%currDir%..\..
set pywinDir=%workingDir%\pywin37
set pysrcDir=%workingDir%\pybibiocr

rem echo %currDir%
rem echo %pywinDir%
rem echo %pysrcDir%


call cd %pysrcDir%
FOR /F "DELIMS==" %%f IN ('DIR "%pysrcDir%\*.ui" /B') DO (
    call %pywinDir%\python.exe -m PyQt5.uic.pyuic %%f -o %%~nf.py
)

call cd %currDir%