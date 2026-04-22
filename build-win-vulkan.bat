@echo off
setlocal enabledelayedexpansion

:: --- Configuration ---
set BUILD_DIR=build
set BUILD_TYPE=Release
set ENABLE_VULKAN=ON

echo ====================================================
echo Qwen3-TTS Distributable Builder (Static + Vulkan)
echo ====================================================

:: 1. Environment Checks
if "%VULKAN_SDK%"=="" (
    echo [ERROR] Vulkan SDK not found.
    pause
    exit /b 1
)

where cl.exe >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Please run from "Developer Command Prompt for VS 2026".
    pause
    exit /b 1
)

:: 2. Prepare Build Dir
if exist %BUILD_DIR% (
    del /q %BUILD_DIR%\CMakeCache.txt >nul 2>&1
) else (
    mkdir %BUILD_DIR%
)

cd %BUILD_DIR%

echo [3/4] Configuring for Maximum Distribution...
:: -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded: Links the C++ runtime into the EXE (No more missing msvcp140.dll)
:: -DGGML_AVX2=OFF: (Optional) Set to OFF if you want to support very old CPUs, otherwise leave ON for speed.
:: -DGGML_STATIC=ON: Ensures GGML logic is bundled where possible.

cmake .. -G "Visual Studio 18 2026" -T v145 ^
      -DGGML_VULKAN=%ENABLE_VULKAN% ^
      -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
      -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" ^
      -DGGML_STATIC=ON ^
      -DCMAKE_POLICY_DEFAULT_CMP0194=NEW

if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configuration failed.
    pause
    exit /b 1
)

:: 4. Build
echo [4/4] Finalizing Build...
cmake --build . --config %BUILD_TYPE% --parallel

echo ====================================================
echo SUCCESS! 
echo Copy the EXE and ALL 4 GGML DLLs to your Subtitle Edit folder.
echo ====================================================
pause