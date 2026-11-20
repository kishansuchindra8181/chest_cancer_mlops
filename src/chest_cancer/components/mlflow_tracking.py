
import json
import logging
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "chest_cancer_model.keras"
METRICS_PATH = PROJECT_ROOT / "data" / "artifacts" / "evaluation_metrics.json"
CONFUSION_PATH = PROJECT_ROOT / "data" / "artifacts" / "confusion_matrix.png"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "artifacts" / "test_predictions.csv"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_tracking():

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())

    mlflow.set_experiment("chest-cancer-detection")

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    with mlflow.start_run(run_name="MobileNetV2-Experiment"):

        # Log model configuration
        mlflow.log_params({
            "model": "MobileNetV2",
            "image_size": "160x160",
            "batch_size": 8,
            "epochs": 8,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "classes": "Bengin,Malignant,Normal",
            "duplicate_handling": "SHA256",
            "split_method": "Stratified image-level split"
        })

        # Log evaluation metrics
        mlflow.log_metric(
            "test_accuracy",
            metrics["test_accuracy"]
        )

        mlflow.log_metric(
            "macro_f1_score",
            metrics["macro_f1_score"]
        )

        # Log model and evaluation artifacts
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(CONFUSION_PATH))
        mlflow.log_artifact(str(PREDICTIONS_PATH))

        logger.info("MLflow experiment logged successfully.")
        logger.info("Experiment: chest-cancer-detection")


if __name__ == "__main__":
    run_tracking()
