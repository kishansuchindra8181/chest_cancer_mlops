
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANIFEST_PATH = PROJECT_ROOT / "data" / "artifacts" / "dataset_manifest.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42


class DataTransformation:

    def run(self):

        logger.info("Starting data transformation...")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load dataset manifest
        df = pd.read_csv(MANIFEST_PATH)

        logger.info("Original dataset size: %d", len(df))

        # 2. Remove exact duplicate images using SHA-256
        df = df.drop_duplicates(subset=["sha256"], keep="first")

        logger.info("Dataset after duplicate removal: %d", len(df))

        # 3. Validate class labels
        expected_classes = {"Bengin", "Malignant", "Normal"}

        actual_classes = set(df["label"].unique())

        if actual_classes != expected_classes:
            raise ValueError(
                f"Expected classes {expected_classes}, got {actual_classes}"
            )

        # 4. Check image paths
        missing_images = df[~df["image_path"].apply(
            lambda path: Path(path).exists()
        )]

        if not missing_images.empty:
            raise FileNotFoundError(
                f"{len(missing_images)} image files are missing."
            )

        # 5. Split into 70% training and 30% temporary data
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=df["label"]
        )

        # 6. Split temporary data equally into validation and testing
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=temp_df["label"]
        )

        # 7. Save split CSV files
        train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)

        val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)

        test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

        df.to_csv(PROCESSED_DIR / "cleaned_manifest.csv", index=False)

        # 8. Display split summary
        logger.info("========== DATA SPLIT SUMMARY ==========")

        logger.info("Training images: %d", len(train_df))
        logger.info("Validation images: %d", len(val_df))
        logger.info("Testing images: %d", len(test_df))

        logger.info("Training distribution:\n%s",
                    train_df["label"].value_counts())

        logger.info("Validation distribution:\n%s",
                    val_df["label"].value_counts())

        logger.info("Testing distribution:\n%s",
                    test_df["label"].value_counts())

        logger.info("Data transformation completed successfully!")


if __name__ == "__main__":

    transformation = DataTransformation()

    transformation.run()
