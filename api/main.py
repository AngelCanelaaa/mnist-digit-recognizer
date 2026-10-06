from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

app = FastAPI(title="API Predicción MNIST")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
try:
    modelo = tf.keras.models.load_model("model/modelo_mnist.keras")
    print("Modelo cargado correctamente")
except Exception as e:
    raise RuntimeError(f"Error cargando el modelo: {e}")

nombre_clases = [str(i) for i in range(10)]

@app.get("/")
def root():
    return {"message": "Modelo MNIST listo y funcionando", "version": "1.0", "clases": nombre_clases}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        # Redimensionar a 28x28 directamente
        image = image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Invertir colores: el modelo espera dígito blanco sobre fondo negro
        # Nuestro frontend envía fondo blanco y trazo negro, así que invertimos.
        img_array = 1.0 - img_array

        # Asegurar forma (1,28,28,1)
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        # DEBUG: Guardar la imagen procesada para inspección (opcional)
        # import cv2
        # cv2.imwrite("debug.png", (img_array[0,:,:,0]*255).astype(np.uint8))

        pred = modelo.predict(img_array, verbose=False)
        clase_idx = int(np.argmax(pred))
        probabilidad = float(np.max(pred))

        return {
            "clase": str(clase_idx),
            "probabilidad": round(probabilidad, 4),
            "todas": {str(i): round(float(pred[0][i]), 4) for i in range(10)}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))