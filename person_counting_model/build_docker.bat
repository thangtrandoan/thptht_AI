@echo off
echo ========================================
echo Building Docker Image for Student Detection System
echo ========================================

REM Set image name and tag
set IMAGE_NAME=moimoi05/person_counting_model-person_counter
set IMAGE_TAG=latest

echo Building Docker image: %IMAGE_NAME%:%IMAGE_TAG%
docker build -t %IMAGE_NAME%:%IMAGE_TAG% .

if %ERRORLEVEL% neq 0 (
    echo ERROR: Docker build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo To push to DockerHub, run:
echo docker push %IMAGE_NAME%:%IMAGE_TAG%
echo.
echo To test locally, run:
echo docker run -p 5000:5000 %IMAGE_NAME%:%IMAGE_TAG%
echo.

pause
