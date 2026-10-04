@echo off
setlocal
if not defined JAVA_HOME (
  echo JAVA_HOME must point to a JDK 17 or newer. 1>&2
  exit /b 1
)
"%JAVA_HOME%\bin\java.exe" -classpath "%~dp0gradle\wrapper\gradle-wrapper.jar" org.gradle.wrapper.GradleWrapperMain %*
