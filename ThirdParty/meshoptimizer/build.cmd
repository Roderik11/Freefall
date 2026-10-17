@echo off
rem Builds win-x64\meshoptimizer.dll from src\ with MSVC (any Visual Studio that has the C++ x64 tools).
rem Static CRT (/MT), so the DLL needs no VC++ redistributable. The engine calls it by P/Invoke
rem (Freefall.Engine\Graphics\MeshOptimizer.cs) and the engine project copies it to the output.
setlocal
set HERE=%~dp0
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSDIR=%%i
if not defined VSDIR (
    echo No Visual Studio with the C++ x64 tools found.
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

set OBJ=%TEMP%\meshoptimizer_obj
if not exist "%OBJ%" mkdir "%OBJ%"
cl /nologo /O2 /MT /LD /W3 /DNDEBUG "/DMESHOPTIMIZER_API=__declspec(dllexport)" /Fo"%OBJ%\\" "%HERE%src\*.cpp" /Fe"%HERE%win-x64\meshoptimizer.dll" /link /NOLOGO /IMPLIB:"%OBJ%\meshoptimizer.lib" || exit /b 1
echo Built %HERE%win-x64\meshoptimizer.dll
