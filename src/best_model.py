from dotenv import load_dotenv

load_dotenv()

import datetime
import os

import mlflow
from mlflow.tracking import MlflowClient
import yaml
from pyspark.ml import PipelineModel

from logger import logging


def _set_tracking_uri():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logging.info("MLflow tracking -> %s", tracking_uri)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        logging.info("MLflow tracking -> sqlite:///mlflow.db")


def _load_best_run_id(client, run_id_path, experiment_name):
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            best_run_id = f.read().strip()
        if best_run_id:
            return best_run_id

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise Exception("MLflow experiment not found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.Test_RMSE ASC"],
        max_results=1,
    )
    if not runs:
        raise Exception("No runs found in MLflow. Run train.py first.")

    best_run_id = runs[0].info.run_id
    os.makedirs(os.path.dirname(run_id_path), exist_ok=True)
    with open(run_id_path, "w") as f:
        f.write(best_run_id)
    return best_run_id


def best_model():
    try:
        logging.info("Starting best model selection process...")

        model_name = "FloodPrediction_Training"
        pipeline_name = "Flood_Transformation_Pipeline"
        experiment_name = "FloodPrediction_Training"
        run_id_path = "logs/best_run_id.txt"

        params = yaml.safe_load(open("params.yaml"))
        pipeline_local_path = params["output"]["pipeline_path"]

        _set_tracking_uri()
        client = MlflowClient()

        best_run_id = _load_best_run_id(client, run_id_path, experiment_name)

        logging.info("===================================")
        logging.info("BEST RUN ID : %s", best_run_id)
        logging.info("===================================")

        model_version = mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name=model_name,
        )
        logging.info("Prediction model registered - version %s", model_version.version)

        best_model = mlflow.spark.load_model(model_uri=f"runs:/{best_run_id}/model")

        best_rmse = client.get_run(best_run_id).data.metrics.get("Test_RMSE", float("inf"))
        best_model_name = client.get_run(best_run_id).data.tags.get("model_type", model_name)

        with mlflow.start_run(run_name="Best_Model_" + best_model_name) as champion_run:
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_metric("best_RMSE", best_rmse)
            mlflow.set_tag("stage", "champion")
            mlflow.set_tag("model_type", best_model_name)
            mlflow.set_tag("framework", "pyspark")
            mlflow.spark.log_model(best_model, artifact_path="best_model")

            if not os.path.exists(pipeline_local_path):
                raise Exception(
                    f"Local pipeline model not found at {pipeline_local_path}. Run preprocessing.py first."
                )

            pipeline_model = PipelineModel.load(pipeline_local_path)
            mlflow.spark.log_model(
                pipeline_model,
                artifact_path="transformation_pipeline",
            )

        pipeline_version = mlflow.register_model(
            model_uri=f"runs:/{champion_run.info.run_id}/transformation_pipeline",
            name=pipeline_name,
        )
        logging.info(
            "Transformation pipeline registered - version %s",
            pipeline_version.version,
        )

        promote_targets = [(model_name, model_version.version), (pipeline_name, pipeline_version.version)]

        for name, version in promote_targets:
            client.transition_model_version_stage(
                name=name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            logging.info("%s v%s -> Production", name, version)

        logging.info("===================================")
        logging.info("Best model selection completed!")
        logging.info("===================================")

        os.makedirs("logs", exist_ok=True)
        with open("logs/best_model.log", "w") as f:
            f.write(f"Best model selection completed at {datetime.datetime.now()}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Pipeline: {pipeline_name}\n")
            logging.info("Log file saved: logs/best_model.log")

    except Exception as e:
        logging.error("Error: %s", str(e))
        raise


if __name__ == "__main__":
    try:
        best_model()
    except Exception as e:
        logging.error("Error: %s", str(e))