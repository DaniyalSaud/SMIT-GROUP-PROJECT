from functools import lru_cache
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import mlflow
import mlflow.spark
import yaml
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

load_dotenv()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("FLOOD_MODEL_NAME", "FloodPrediction_Training")
PIPELINE_NAME = os.getenv("FLOOD_PIPELINE_NAME", "Flood_Transformation_Pipeline")
MODEL_STAGE = os.getenv("FLOOD_MODEL_STAGE", "Production")
PIPELINE_STAGE = os.getenv("FLOOD_PIPELINE_STAGE", "Production")

with open("params.yaml", "r", encoding="utf-8") as params_file:
    PARAMS = yaml.safe_load(params_file)

LOCAL_PIPELINE_PATH = PARAMS["output"]["pipeline_path"]

FEATURE_NAMES = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "WetlandLoss",
]


class FloodPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    MonsoonIntensity: float
    TopographyDrainage: float
    RiverManagement: float
    Deforestation: float
    Urbanization: float
    ClimateChange: float
    DamsQuality: float
    Siltation: float
    AgriculturalPractices: float
    Encroachments: float
    IneffectiveDisasterPreparedness: float
    DrainageSystems: float
    CoastalVulnerability: float
    Landslides: float
    Watersheds: float
    DeterioratingInfrastructure: float
    WetlandLoss: float


class FloodPredictionResponse(BaseModel):
    status: str
    model_name: str
    pipeline_name: str
    predicted_flood_probability: float


def _configure_mlflow() -> None:
    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

_configure_mlflow()

app = FastAPI(
    title="Flood Prediction API",
    version="1.0.0",
    description="Flood prediction using Spark MLlib + MLflow",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

spark = SparkSession.builder.appName("FloodPredictionAPI").getOrCreate()

feature_assembler = VectorAssembler(inputCols=FEATURE_NAMES, outputCol="features")


@lru_cache(maxsize=1)
def _load_artifacts():
    model = mlflow.spark.load_model(
        model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    )
    try:
        pipeline_model = mlflow.spark.load_model(
            model_uri=f"models:/{PIPELINE_NAME}/{PIPELINE_STAGE}"
        )
    except Exception:
        pipeline_model = PipelineModel.load(LOCAL_PIPELINE_PATH)
    return model, pipeline_model


@app.on_event("startup")
def startup_event():
    _load_artifacts()


@app.get("/")
def home():
    return {"message": "Flood Prediction API Running"}


@app.get("/tester")
def tester():
    return {"message": "tester endpoint is working fine"}


@app.post("/predict", response_model=FloodPredictionResponse)
def predict(payload: FloodPredictionRequest):
    try:
        model, pipeline_model = _load_artifacts()
        input_df = spark.createDataFrame([payload.model_dump()])

        transformed_df = pipeline_model.transform(input_df)
        prepared_df = transformed_df.select(
            *[
                vector_to_array("features_scaled")[index].alias(feature_name)
                for index, feature_name in enumerate(FEATURE_NAMES)
            ]
        )
        prepared_df = feature_assembler.transform(prepared_df)

        prediction_df = model.transform(prepared_df)
        prediction = prediction_df.select("prediction").collect()[0][0]

        return FloodPredictionResponse(
            status="success",
            model_name=MODEL_NAME,
            pipeline_name=PIPELINE_NAME,
            predicted_flood_probability=round(float(prediction), 4),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc