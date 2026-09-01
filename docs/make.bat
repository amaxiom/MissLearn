@ECHO OFF

REM Sphinx build helper for Windows, which is where this package is developed.
REM
REM `python` on PATH has no scientific stack on the development machine, so
REM SPHINXBUILD defaults to calling sphinx through the Anaconda interpreter.
REM Override it if your environment differs:
REM
REM     set SPHINXBUILD=sphinx-build
REM     make strict

pushd %~dp0

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=C:\ProgramData\anaconda3\python.exe -m sphinx
)
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo Could not run sphinx-build. Set the SPHINXBUILD environment variable
	echo to point at the interpreter or launcher that has Sphinx installed,
	echo or install it with:
	echo.
	echo     pip install -r requirements.txt
	echo.
	exit /b 1
)

if "%1" == "" goto help
if "%1" == "strict" goto strict
if "%1" == "clean" goto clean

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:strict
REM Exactly what CI runs. Use this before pushing.
%SPHINXBUILD% -b html -W --keep-going %SOURCEDIR% %BUILDDIR%\html %SPHINXOPTS% %O%
goto end

:clean
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
if exist generated rmdir /s /q generated
if exist auto_examples rmdir /s /q auto_examples
if exist sg_execution_times.rst del /q sg_execution_times.rst
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
echo.
echo   strict      build exactly as CI does, warnings treated as errors
echo   clean       remove _build, and the generated API and gallery trees

:end
popd
