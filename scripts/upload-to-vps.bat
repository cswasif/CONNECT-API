@echo off
REM Upload ConnectAPI to VPS
REM Run this from Windows PowerShell or Command Prompt

echo ========================================
echo ConnectAPI - Upload to VPS
echo ========================================

set SSH_KEY=c:\Users\Administrator\Desktop\Oracle\md.wasif.faisal@g.bracu.ac.bd-2025-11-21T09_12_08.635Z.pem
set VPS_USER=ubuntu
set VPS_HOST=213.35.103.204
set LOCAL_DIR=c:\Users\Administrator\Desktop\Oracle\zconnapi2-main
set REMOTE_DIR=/home/ubuntu/

echo.
echo Uploading files to VPS...
echo.

scp -i "%SSH_KEY%" -r "%LOCAL_DIR%" %VPS_USER%@%VPS_HOST%:%REMOTE_DIR%

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Upload completed successfully!
    echo ========================================
    echo.
    echo Next steps:
    echo 1. SSH into VPS:
    echo    ssh -i "%SSH_KEY%" %VPS_USER%@%VPS_HOST%
    echo.
    echo 2. Navigate to directory:
    echo    cd /home/ubuntu/zconnapi2-main
    echo.
    echo 3. Run setup:
    echo    chmod +x install-redis.sh setup.sh deploy.sh
    echo    ./install-redis.sh
    echo    ./setup.sh
    echo.
    echo 4. See DEPLOYMENT.md for complete guide
    echo.
) else (
    echo.
    echo ========================================
    echo Upload failed!
    echo ========================================
    echo Please check your connection and try again.
)

pause
