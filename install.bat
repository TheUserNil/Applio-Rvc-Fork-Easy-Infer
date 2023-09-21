@echo off
setlocal

set "repoUrl=https://github.com/IAHispano/Applio-RVC-Fork.git"
set "repoFolder=Applio-RVC-Fork"
set "principal=%cd%\%repoFolder%"
set "runtime_scripts=%cd%\%repoFolder%\runtime\Scripts"
set "URL_BASE=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
set "URL_EXTRA=https://huggingface.co/IAHispano/applio/resolve/main"

cls
for /f "delims=: tokens=*" %%A in ('findstr /b ":::" "%~f0"') do @echo(%%A
echo.

cd "assets"

cd "rmvpe"
curl -LJO "%URL_BASE%/rmvpe.pt"
echo.
cls
cd ".."

cd "hubert"
curl -LJO "%URL_BASE%/hubert_base.pt"
echo.
cls

cls 
echo Applio has been successfully downloaded, run the file go-applio.bat to run the web interface!
echo.
pause
exit