
@echo off
set currDir=%~dp0
set workingDir=%currDir%\..\..
set pywinDir=%workingDir%\pywin37

FOR /F "DELIMS==" %%f IN ('DIR "%currDir%\*.ui" /B') DO (
    call %pywinDir%\python.exe -m PyQt5.uic.pyuic %%f -o %%~nf.py
)