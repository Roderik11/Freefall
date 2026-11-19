@echo off
rem Builds win-x64\freefall_dlss.dll, the engine's DLSS shim (src\freefall_dlss.cpp), with MSVC (any Visual Studio that
rem has the C++ x64 tools) against NVIDIA's NGX SDK in sdk\ (fetched by fetch.ps1, not committed). Static CRT (/MT), as
rem nvsdk_ngx_s.lib is built, so the DLL needs no VC++ redistributable. The engine calls it by P/Invoke
rem (Freefall.Engine\Graphics\Upscaling\Dlss.cs) and the engine project copies it and nvngx_dlss.dll to the output.
setlocal
set HERE=%~dp0
if not exist "%HERE%sdk\lib\nvsdk_ngx_s.lib" (
    echo NGX SDK missing: run fetch.ps1 first.
    exit /b 1
)
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSDIR=%%i
if not defined VSDIR (
    echo No Visual Studio with the C++ x64 tools found.
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

set OBJ=%TEMP%\freefall_dlss_obj
if not exist "%OBJ%" mkdir "%OBJ%"
if not exist "%HERE%win-x64" mkdir "%HERE%win-x64"
cl /nologo /O2 /MT /LD /EHsc /W3 /DNDEBUG /I"%HERE%sdk\include" /Fo"%OBJ%\\" "%HERE%src\freefall_dlss.cpp" /Fe"%HERE%win-x64\freefall_dlss.dll" /link /NOLOGO /IMPLIB:"%OBJ%\freefall_dlss.lib" "%HERE%sdk\lib\nvsdk_ngx_s.lib" d3d12.lib advapi32.lib user32.lib || exit /b 1
echo Built %HERE%win-x64\freefall_dlss.dll
