@echo off
REM Launch Streamlit App (Demo)

SET PYTHON_PATH=C:\Users\Admin\anaconda3\envs\creating_an_environment\python.exe
SET APP_PATH=C:\Users\Admin\Documents\GitHub\Unsupervised_Learning_Project\app.py

cd /d C:\Users\Admin\Documents\GitHub\Unsupervised_Learning_Project

"%PYTHON_PATH%" -m streamlit run "%APP_PATH%"

pause