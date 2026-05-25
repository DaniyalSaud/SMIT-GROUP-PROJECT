# Flood Prediction Project

This repository contains a Spark + MLflow flood prediction pipeline, a FastAPI inference service, and a Streamlit UI.

## Run the backend

Start the FastAPI app from the project root:

```bash
python main.py
```

On Windows, prefer `python main.py` or `uvicorn app.main:app --reload` instead of `fastapi dev`.
The FastAPI CLI banner currently uses a rocket emoji that can fail on some console encodings.

The API serves:

- `GET /` for a basic health message
- `GET /tester` for a simple smoke test
- `POST /predict` for flood inference using the 17-feature JSON payload

## Run the UI

Launch the Streamlit interface in a second terminal:

```bash
streamlit run ui/app.py
```

If your backend is not running on `http://127.0.0.1:8000`, set `FLOOD_API_URL` before starting Streamlit.

## Required environment variables

- `MLFLOW_TRACKING_URI` for MLflow model resolution
- `FLOOD_MODEL_NAME`, `FLOOD_PIPELINE_NAME`, `FLOOD_MODEL_STAGE`, and `FLOOD_PIPELINE_STAGE` if you want to override the defaults

## Prediction payload

Send a JSON body with these fields:

`MonsoonIntensity`, `TopographyDrainage`, `RiverManagement`, `Deforestation`, `Urbanization`, `ClimateChange`, `DamsQuality`, `Siltation`, `AgriculturalPractices`, `Encroachments`, `IneffectiveDisasterPreparedness`, `DrainageSystems`, `CoastalVulnerability`, `Landslides`, `Watersheds`, `DeterioratingInfrastructure`, `WetlandLoss`
