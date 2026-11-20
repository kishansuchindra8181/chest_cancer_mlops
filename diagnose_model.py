
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "chest_cancer_model.keras"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"

CLASS_NAMES = ["Bengin", "Malignant", "Normal"]
IMAGE_SIZE = (160, 160)


def preprocess_image(path):
    image = tf.io.read_file(str(path))
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return tf.expand_dims(image, axis=0)


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    df = pd.read_csv(TEST_CSV)

    print("MODEL:", MODEL_PATH)
    print("OUTPUT SHAPE:", model.output_shape)

    for class_name in CLASS_NAMES:
        sample = df[df["label"] == class_name].head(1)

        if sample.empty:
            continue

        row = sample.iloc[0]
        image_path = ROOT / row["image_path"]

        probabilities = model.predict(
            preprocess_image(image_path),
            verbose=0
        )[0]

        predicted_index = int(np.argmax(probabilities))

        print("\n------------------------------")
        print("IMAGE:", row["filename"])
        print("ACTUAL CLASS:", class_name)
        print("PREDICTED CLASS:", CLASS_NAMES[predicted_index])

        for i, name in enumerate(CLASS_NAMES):
            print(f"{name}: {probabilities[i] * 100:.2f}%")


if __name__ == "__main__":
    main()
