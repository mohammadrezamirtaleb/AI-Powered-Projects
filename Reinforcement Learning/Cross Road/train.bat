@echo off
title Autonomous Crossroad Deep RL Fast Headless Trainer
echo =========================================================
echo   Starting High-Speed Deep RL Training for Crossroad AI
echo =========================================================
call venv\Scripts\activate.bat
python src\train_headless.py
pause
