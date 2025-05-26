@echo off
echo ========================================
echo Pushing Docker Image to DockerHub
echo ========================================

REM Set image name and tag
set IMAGE_NAME=moimoi05/person_counting_model-person_counter
set IMAGE_TAG=latest

echo Checking if you're logged in to DockerHub...
docker info > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Docker is not running!
    pause
    exit /b 1
)

echo.
echo Pushing image: %IMAGE_NAME%:%IMAGE_TAG%
echo This may take several minutes...
echo.

docker push %IMAGE_NAME%:%IMAGE_TAG%

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Docker push failed!
    echo Make sure you're logged in to DockerHub:
    echo docker login
    pause
    exit /b 1
)

echo.
echo ========================================
echo Push completed successfully!
echo ========================================
echo.
echo Image is now available at:
echo https://hub.docker.com/r/moimoi05/person_counting_model-person_counter
echo.
echo Teachers can now use:
echo docker pull %IMAGE_NAME%:%IMAGE_TAG%
echo.

pause
