
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "chest_cancer_model.keras"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "artifacts"

IMAGE_SIZE = (160, 160)

CLASS_NAMES = ["Bengin", "Malignant", "Normal"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(path):

    image = tf.io.read_file(str(path))
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image


class ModelEvaluation:

    def run(self):

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Loading trained model...")

        model = tf.keras.models.load_model(MODEL_PATH)

        test_df = pd.read_csv(TEST_PATH)

        label_to_index = {
            name: index for index, name in enumerate(CLASS_NAMES)
        }

        y_true = test_df["label"].map(label_to_index).values

        images = np.array([
            preprocess_image(path).numpy()
            for path in test_df["image_path"]
        ])

        logger.info("Generating predictions...")

        probabilities = model.predict(images, batch_size=8)

        y_pred = np.argmax(probabilities, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro"
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2]
        )

        # Save metrics JSON
        metrics = {
            "test_accuracy": float(accuracy),
            "macro_f1_score": float(macro_f1),
            "classification_report": report,
            "confusion_matrix": matrix.tolist()
        }

        with open(
            OUTPUT_DIR / "evaluation_metrics.json",
            "w"
        ) as file:
            json.dump(metrics, file, indent=4)

        # Save confusion matrix image
        fig, ax = plt.subplots(figsize=(7, 6))

        display = ax.imshow(matrix, cmap="Blues")

        ax.set_title("Chest Cancer Detection - Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))

        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)

        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center"
                )

        fig.colorbar(display)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "confusion_matrix.png",
            dpi=150
        )

        plt.close(fig)

        # Save prediction results
        results = test_df.copy()

        results["actual_label"] = [
            CLASS_NAMES[index] for index in y_true
        ]

        results["predicted_label"] = [
            CLASS_NAMES[index] for index in y_pred
        ]

        results["confidence"] = np.max(
            probabilities,
            axis=1
        )

        results.to_csv(
            OUTPUT_DIR / "test_predictions.csv",
            index=False
        )

        logger.info("========== EVALUATION RESULTS ==========")

        logger.info("Test Accuracy: %.4f", accuracy)
        logger.info("Macro F1 Score: %.4f", macro_f1)

        logger.info(
            "\n%s",
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                target_names=CLASS_NAMES,
                zero_division=0
            )
        )

        logger.info("Evaluation completed successfully!")


if __name__ == "__main__":

    evaluator = ModelEvaluation()
    evaluator.run()
