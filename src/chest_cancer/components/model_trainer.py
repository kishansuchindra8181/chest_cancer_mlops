
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 8

CLASS_NAMES = ["Bengin", "Malignant", "Normal"]

CLASS_TO_INDEX = {
    name: index for index, name in enumerate(CLASS_NAMES)
}


def load_image(path, label):

    image = tf.io.read_file(path)

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

    return image, label


def create_dataset(dataframe, shuffle=False):

    paths = dataframe["image_path"].astype(str).values

    labels = dataframe["label"].map(CLASS_TO_INDEX).values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


class ModelTrainer:

    def run(self):

        logger.info("Loading processed datasets...")

        train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
        val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
        test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Create TensorFlow datasets
        train_ds = create_dataset(train_df, shuffle=True)
        val_ds = create_dataset(val_df)
        test_ds = create_dataset(test_df)

        # Calculate class weights
        train_labels = train_df["label"].map(
            CLASS_TO_INDEX
        ).values

        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(CLASS_NAMES)),
            y=train_labels
        )

        class_weights = {
            i: float(weight)
            for i, weight in enumerate(weights)
        }

        logger.info("Class weights: %s", class_weights)

        # Load pretrained MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(160, 160, 3),
            include_top=False,
            weights="imagenet"
        )

        # Freeze pretrained layers
        base_model.trainable = False

        # Build classification model
        inputs = tf.keras.Input(shape=(160, 160, 3))

        x = base_model(inputs, training=False)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(
            128,
            activation="relu"
        )(x)

        outputs = tf.keras.layers.Dense(
            3,
            activation="softmax"
        )(x)

        model = tf.keras.Model(inputs, outputs)

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.summary()

        # Training callbacks
        callbacks = [

            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2
            ),

            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(
                    MODEL_DIR / "best_model.keras"
                ),
                monitor="val_loss",
                save_best_only=True
            )
        ]

        logger.info("Starting model training...")

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks
        )

        # Load best validation model
        best_model = tf.keras.models.load_model(
            MODEL_DIR / "best_model.keras"
        )

        # Evaluate on held-out test data
        test_loss, test_accuracy = best_model.evaluate(
            test_ds
        )

        logger.info(
            "Test Loss: %.4f",
            test_loss
        )

        logger.info(
            "Test Accuracy: %.4f",
            test_accuracy
        )

        # Save final model
        best_model.save(
            MODEL_DIR / "chest_cancer_model.keras"
        )

        # Save training history
        pd.DataFrame(history.history).to_csv(
            MODEL_DIR / "training_history.csv",
            index=False
        )

        logger.info("Model saved successfully!")

        logger.info(
            "Model path: %s",
            MODEL_DIR / "chest_cancer_model.keras"
        )


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.run()
