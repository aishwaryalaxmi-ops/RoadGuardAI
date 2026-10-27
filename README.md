# RoadGuard AI
AI-Powered Traffic Intelligence & Road Safety Prediction Platform

## Important
The included demo-data generator exists only to test the application end-to-end. Do **not** use its model scores as research accuracy. Replace `data/traffic_accident_data.csv` with a real, representative dataset before reporting results.

## Run in VS Code
1. Open this folder in VS Code.
2. Create and activate a virtual environment.
3. Install requirements.
4. Put your real CSV at `data/traffic_accident_data.csv` using the schema in `data/DATA_SCHEMA.md`.
5. Run `python src/train_models.py`.
6. Run `streamlit run app.py`.

## Windows commands
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/generate_demo_data.py
python src/train_models.py
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:
```cmd
.venv\Scripts\activate.bat
```
