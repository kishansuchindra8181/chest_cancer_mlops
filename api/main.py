
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "chest_cancer_savedmodel"

# Model configuration
CLASS_NAMES = ["Bengin", "Malignant", "Normal"]
IMAGE_SIZE = (160, 160)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Initialize FastAPI
app = FastAPI(
    title="Chest Cancer Classification API",
    description="Chest cancer image classification using deep learning",
    version="1.0.0"
)

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


@app.get("/")
def home():
    return {
        "message": "Chest Cancer Classification API is running",
        "status": "success"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are allowed"
        )

    try:
        # Read uploaded image
        contents = await file.read()

        # Validate file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size must not exceed 10 MB"
            )

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # IMPORTANT:
        # Use exactly the same preprocessing as model training
        image_array = tf.io.decode_image(
            contents,
            channels=3,
            expand_animations=False
        )

        image_array.set_shape([None, None, 3])

        image_array = tf.image.resize(
            image_array,
            IMAGE_SIZE
        )

        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )

        image_array = tf.expand_dims(image_array, axis=0)

        # Generate prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(tf.argmax(predictions).numpy())
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])

        probabilities = {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }

        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "probabilities": {
                    name: round(probability * 100, 2)
                    for name, probability in probabilities.items()
                },
                "disclaimer": (
                    "This is an experimental AI prediction, "
                    "not a medical diagnosis. Consult a qualified doctor."
                )
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        await file.close()
