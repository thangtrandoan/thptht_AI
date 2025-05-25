@echo off
REM This script creates a Python 3.10 virtual environment and installs required packages

echo Creating virtual environment with Python 3.10...
python -m venv venv_py310 --prompt="person_counter"

echo Activating virtual environment...
call venv_py310\Scripts\activate.bat

echo Installing required packages...
pip install -r requirements.txt

echo Virtual environment setup complete!
echo To activate the virtual environment in the future, run: venv_py310\Scripts\activate.bat
echo To deactivate the virtual environment, run: deactivate

pause
