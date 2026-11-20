
import os
import json
import hashlib
import logging
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "The IQ-OTHNCCD lung cancer dataset"
)

ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

MANIFEST_PATH = ARTIFACT_DIR / "dataset_manifest.csv"
REPORT_PATH = ARTIFACT_DIR / "ingestion_report.json"

CLASS_MAPPING = {
    "bengin cases": "Bengin",
    "benign cases": "Benign",
    "malignant cases": "Malignant",
    "normal cases": "Normal",
}

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------

def calculate_hash(file_path):
    """Calculate SHA256 hash for duplicate detection."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def validate_image(file_path):
    """Check whether an image can be opened and decoded."""

    try:
        with Image.open(file_path) as image:
            image.verify()

        # Reopen to check actual image dimensions.
        with Image.open(file_path) as image:
            width, height = image.size
            image_format = image.format

        if width <= 0 or height <= 0:
            return False, "Invalid image dimensions", None, None, None

        return True, None, width, height, image_format

    except (OSError, ValueError, UnidentifiedImageError) as error:
        return False, str(error), None, None, None


# -------------------------------------------------
# DATA INGESTION
# -------------------------------------------------

class DataIngestion:

    def __init__(self):
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):

        logger.info("Starting dataset ingestion...")

        if not DATASET_DIR.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {DATASET_DIR}"
            )

        records = []
        corrupted_images = []
        unsupported_files = []
        hash_groups = defaultdict(list)

        class_counts = Counter()

        class_folders = [
            folder
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir()
        ]

        logger.info(
            "Detected %d class directories",
            len(class_folders)
        )

        # Validate expected classes.
        detected_labels = {
            CLASS_MAPPING[folder.name.strip().lower()]
            for folder in class_folders
            if folder.name.strip().lower() in CLASS_MAPPING
        }

        expected_labels = {"Bengin", "Malignant", "Normal"}

        missing_classes = expected_labels - detected_labels

        if missing_classes:
            raise ValueError(
                f"Missing dataset classes: {missing_classes}"
            )

        # Collect image paths.
        all_files = []

        for folder in class_folders:

            label = CLASS_MAPPING.get(folder.name.strip().lower())

            if label is None:
                logger.warning(
                    "Unknown class folder skipped: %s",
                    folder.name
                )
                continue

            files = [
                file
                for file in folder.rglob("*")
                if file.is_file()
            ]

            for file in files:
                all_files.append((file, label))

        logger.info("Total files detected: %d", len(all_files))

        # Process each image.
        for file_path, label in tqdm(
            all_files,
            desc="Validating images"
        ):

            if file_path.suffix.lower() not in SUPPORTED_FORMATS:
                unsupported_files.append(str(file_path))
                continue

            is_valid, error, width, height, image_format = (
                validate_image(file_path)
            )

            if not is_valid:

                corrupted_images.append({
                    "file": str(file_path),
                    "error": error
                })

                logger.warning("Corrupted image: %s", file_path)
                continue

            file_hash = calculate_hash(file_path)

            class_counts[label] += 1

            record = {
                "image_path": str(file_path.relative_to(PROJECT_ROOT)),
                "filename": file_path.name,
                "label": label,
                "width": width,
                "height": height,
                "format": image_format,
                "file_size_kb": round(
                    file_path.stat().st_size / 1024, 2
                ),
                "sha256": file_hash
            }

            records.append(record)

            hash_groups[file_hash].append(record)

        # Identify duplicates without deleting anything.
        duplicate_groups = []

        for file_hash, group in hash_groups.items():

            if len(group) > 1:

                duplicate_groups.append({
                    "sha256": file_hash,
                    "files": [
                        item["image_path"] for item in group
                    ],
                    "labels": list(set(
                        item["label"] for item in group
                    )),
                    "cross_class_conflict": (
                        len(set(item["label"] for item in group)) > 1
                    )
                })

        # Save image manifest.
        dataframe = pd.DataFrame(records)

        dataframe.to_csv(
            MANIFEST_PATH,
            index=False
        )

        # Prepare report.
        report = {
            "dataset": "IQ-OTH/NCCD Lung Cancer Dataset",
            "total_files_detected": len(all_files),
            "valid_images": len(records),
            "corrupted_images": len(corrupted_images),
            "unsupported_files": len(unsupported_files),
            "duplicate_groups": len(duplicate_groups),
            "cross_class_duplicate_groups": sum(
                item["cross_class_conflict"]
                for item in duplicate_groups
            ),
            "class_distribution": dict(class_counts),
            "corrupted_image_details": corrupted_images,
            "unsupported_file_details": unsupported_files,
            "duplicate_details": duplicate_groups,
            "manifest_path": str(MANIFEST_PATH)
        }

        with open(REPORT_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        # Display summary.
        logger.info("========== INGESTION SUMMARY ==========")

        logger.info("Total files: %d", len(all_files))
        logger.info("Valid images: %d", len(records))
        logger.info("Corrupted images: %d", len(corrupted_images))
        logger.info("Unsupported files: %d", len(unsupported_files))

        logger.info("Class distribution:")

        for class_name, count in class_counts.items():
            logger.info("%s: %d", class_name, count)

        logger.info("Manifest saved: %s", MANIFEST_PATH)
        logger.info("Report saved: %s", REPORT_PATH)

        logger.info("Dataset ingestion completed!")

        return report


# -------------------------------------------------
# EXECUTION
# -------------------------------------------------

if __name__ == "__main__":

    ingestion = DataIngestion()

    ingestion.run()
