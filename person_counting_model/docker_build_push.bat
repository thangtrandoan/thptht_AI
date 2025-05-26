@echo off
echo ========================================
echo Docker Build and Push Script
echo Student Detection System
echo ========================================

REM Set image name and tag
set IMAGE_NAME=moimoi05/person_counting_model-person_counter
set IMAGE_TAG=latest

echo.
echo Building Docker image...
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
set /p PUSH_CHOICE="Do you want to push to DockerHub? (y/n): "
if /i "%PUSH_CHOICE%"=="y" (
    echo.
    echo Pushing to DockerHub...
    docker push %IMAGE_NAME%:%IMAGE_TAG%
    
    if %ERRORLEVEL% neq 0 (
        echo.
        echo ERROR: Docker push failed!
        echo Make sure you're logged in: docker login
        pause
        exit /b 1
    )
    
    echo.
    echo ========================================
    echo Successfully pushed to DockerHub!
    echo ========================================
    echo.
    echo Image URL: https://hub.docker.com/r/moimoi05/person_counting_model-person_counter
) else (
    echo.
    echo Skipping push to DockerHub.
    echo To push later, run: docker push %IMAGE_NAME%:%IMAGE_TAG%
)

echo.
echo ========================================
echo Process completed!
echo ========================================
echo.
echo To test locally:
echo docker run -p 5000:5000 -v "%CD%\students_list.json:/app/students_list.json" -v "%CD%\known_student_faces:/app/known_student_faces" %IMAGE_NAME%:%IMAGE_TAG%
echo.

pause
