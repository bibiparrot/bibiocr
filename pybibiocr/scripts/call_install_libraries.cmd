@echo off
set currDir=%~dp0
set workingDir=%currDir%..\..
set pywinDir=%workingDir%\pywin37
set pySrc=%workingDir%\python
echo %pywinDir%
REM  pyautogui installation error:  AttributeError: module 'enum' has no attribute 'IntFlag'
REM  call %condaRoot%\Scripts\pip.exe install -U --pre uiautomator2
REM  call %condaRoot%\Scripts\pip.exe  uninstall enum34
call %pywinDir%\Scripts\pip.exe install -r %pySrc%\requirements.txt