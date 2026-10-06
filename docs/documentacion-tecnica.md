## DOCUMENTACIÓN TÉCNICA

Proyecto MNIST — Clasificación de Dígitos con IA

FastAPI · TensorFlow · Keras · Docker · Frontend Canvas

**Angel de Jesús Canela Figueroa**

Abril 2026

Mac M3 (Apple Silicon) — macOS

---

## 1. Descripción del Proyecto

Este proyecto implementa un sistema completo de reconocimiento de dígitos manuscritos usando una red neuronal entrenada con el dataset MNIST. El sistema consta de tres componentes principales:

- Backend: API REST construida con FastAPI y Python que carga el modelo Keras y expone endpoints para hacer predicciones.
- Contenedor Docker: La API se ejecuta dentro de un contenedor Docker con plataforma linux/amd64 para compatibilidad con Mac M3 (Apple Silicon).
- Frontend: Página web con canvas interactivo donde el usuario dibuja un dígito del 0 al 9 y el modelo predice en tiempo real qué número es.

El flujo completo es: el usuario dibuja en el navegador → el frontend convierte el canvas a imagen PNG → la envía a la API en Docker → el modelo Keras hace la predicción → el resultado se muestra en pantalla con porcentaje de confianza.

---

## 2. Estructura del Proyecto

El proyecto se organizó con la siguiente estructura de carpetas:

```bash
mnist-digit-recognizer/

├── venv/ # Entorno virtual Python 3.12

├── api/

│ ├── model/

│ │ └── modelo_mnist.keras # Modelo entrenado

│ ├── main.py # API FastAPI

│ ├── requirements.txt # Dependencias

│ └── Dockerfile # Configuración Docker

└── frontend/

└── index.html # Canvas para dibujar
```

---

## 3. Herramientas y Versiones

| **Herramienta**    | **Versión** | **Propósito**                       |
|--------------------|-------------|-------------------------------------|
| Python             | 3.12.13     | Lenguaje base del proyecto          |
| TensorFlow / Keras | 2.20.0      | Cargar y ejecutar el modelo de IA   |
| FastAPI            | 0.115.0     | Framework para crear la API REST    |
| Uvicorn            | 0.30.6      | Servidor ASGI para correr FastAPI   |
| Pillow             | latest      | Procesamiento de imágenes en Python |
| NumPy              | latest      | Operaciones matemáticas con arrays  |
| Docker Desktop     | latest      | Contenedorización de la API         |
| pyenv              | 2.6.27      | Gestión de versiones de Python      |
| Homebrew           | latest      | Gestor de paquetes macOS            |

---

## 4. Proceso de Implementación — Paso a Paso

### 4.1. Instalación de Python 3.12 con pyenv

El modelo fue entrenado en Google Colab con Python 3.12.13 y TensorFlow 2.20.0. Para garantizar compatibilidad, se necesitaba la misma versión en local. pyenv estaba instalado pero desactualizado, por lo que primero se actualizó:

```bash
brew update && brew upgrade pyenv
```

Después se instaló Python 3.12.13:

```bash
pyenv install 3.12.13
```

Para activar esta versión en el proyecto se inicializó pyenv correctamente (Anaconda sobreescribía la versión por defecto):

```bash
eval "$(pyenv init -)"

pyenv local 3.12.13

python --version # Python 3.12.13
```

### 4.2. Creación de la Estructura de Carpetas

Se creó el directorio del proyecto con su estructura completa desde la terminal:

```bash
cd ~/Documents

mkdir mnist-digit-recognizer

cd mnist-digit-recognizer

mkdir -p api/model

mkdir frontend
```

El modelo entrenado se colocó en la carpeta correcta:

```bash
mv ~/Downloads/modelo_mnist.keras ~/Documents/mnist-digit-recognizer/api/model/
```

### 4.3. Creación del Entorno Virtual

Se creó un entorno virtual Python aislado para que las dependencias no conflictúen con otros proyectos o con el Python global del sistema:

```bash
python -m venv venv

source venv/bin/activate
```

La activación se confirmó al ver (venv) al inicio del prompt de la terminal.

* En Mac M3 con Anaconda instalado, el prompt muestra ((venv)) (base). Esto es normal — el (base) es Anaconda corriendo en segundo plano y no afecta el funcionamiento.*

### 4.4. Archivo requirements.txt

Se creó el archivo de dependencias dentro de api/ con las versiones específicas. Nota importante: tensorflow-cpu no existe para arquitectura ARM (Apple Silicon M1/M2/M3), por lo que se usa tensorflow para el entorno local:

```bash
fastapi==0.115.0

uvicorn[standard]==0.30.6

tensorflow==2.20.0 # Para Mac M3 — NO usar tensorflow-cpu

numpy

pydantic==2.7.4

pillow

python-multipart
```

Instalación de dependencias:

```bash
pip install -r api/requirements.txt
```

Verificación de que TensorFlow instaló correctamente:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"

# Resultado: 2.20.0
```

### 4.5. Archivo main.py — API FastAPI

Se creó el archivo principal de la API dentro de api/. Este archivo contiene tres componentes clave:

**a) Inicialización de la app y CORS**

Se configuró CORS (Cross-Origin Resource Sharing) para permitir que el frontend HTML pueda comunicarse con la API desde el navegador:

```bash
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

**b) Carga del modelo al arrancar**

El modelo se carga una sola vez cuando la API inicia, no en cada petición. Esto es importante para el rendimiento:

```bash
modelo = tf.keras.models.load_model("model/modelo_mnist.keras")
```

**c) Endpoint /predict con preprocesamiento mejorado**

El endpoint recibe una imagen PNG, la procesa y devuelve la predicción. Se implementó un preprocesamiento avanzado para mejorar la precisión con imágenes dibujadas a mano:

- Umbralización: convierte píxeles grises (antialiasing del canvas) a blanco o negro puro, eliminando ruido.
- Bounding box: detecta exactamente dónde está el número dibujado y recorta solo esa área, igual que MNIST.
- Padding del 20%: agrega espacio alrededor del número para imitar el estilo del dataset MNIST.
- Cuadrado: si el número quedó rectangular (como el 1), lo hace cuadrado para evitar distorsión al redimensionar.
- Resize LANCZOS: algoritmo de alta calidad para reducir de 360x360 a 28x28 píxeles preservando detalles.
- Detección de fondo: invierte colores automáticamente si detecta fondo blanco, ya que MNIST usa fondo negro.

La respuesta incluye la clase predicha, la probabilidad de confianza y la distribución completa de probabilidades para los 10 dígitos:

```bash
return { "clase": str(clase_idx), "probabilidad": round(probabilidad, 4),

"todas": {str(i): round(float(pred[0][i]), 4) for i in range(10)} }
```

### 4.6. Prueba Local con Uvicorn

Antes de Docker, se probó la API corriendo directamente con Uvicorn para verificar que el modelo cargaba y los endpoints respondían:

```bash
cd api

uvicorn main:app --reload
```

Se verificó en el navegador en:

```bash
http://127.0.0.1:8000 # Health check

http://127.0.0.1:8000/docs # Swagger UI automática
```

Resultado confirmado:

```bash
 Modelo cargado correctamente

INFO: Uvicorn running on http://127.0.0.1:8000
```

### 4.7. Dockerfile — Configuración para Mac M3

Se creó el Dockerfile con ajustes específicos para Mac M3. La diferencia crítica respecto a la guía original es la línea FROM con la plataforma explícita:

```bash
FROM --platform=linux/amd64 python:3.12-slim
```

Sin esta línea, Docker en Mac M3 intenta construir para ARM64 pero tensorflow-cpu dentro del contenedor solo existe para AMD64/Intel, lo que causaría un error. Con linux/amd64 forzado, Docker usa emulación Rosetta para correr la imagen correctamente.

El Dockerfile también instala dependencias del sistema necesarias para Pillow y TensorFlow (gcc, libglib2.0-0, libsm6, libxext6, libxrender-dev) y usa tensorflow-cpu dentro del contenedor porque linux/amd64 sí lo soporta, a diferencia del entorno local Mac.

* La advertencia amarilla en VSCode sobre --platform es solo informativa. El comando es válido y necesario para Mac M3.*

### 4.8. Build y Ejecución del Contenedor Docker

Se construyó la imagen Docker especificando la plataforma para Mac M3:

```bash
docker buildx build --platform linux/amd64 -t img-api-mnist .
```

Se creó y ejecutó el contenedor mapeando el puerto 8000:

```bash
docker run -d --name srv-api-mnist -p 8000:8000 img-api-mnist:latest
```

Verificación de que el contenedor está corriendo:

```bash
docker ps
```

Resultado:

```bash
CONTAINER ID IMAGE STATUS PORTS

3a850a864718 img-api-mnist:latest Up X minutes 0.0.0.0:8000->8000/tcp
```

Verificación en el navegador:

```bash
http://localhost:8000

# {"message":"Modelo MNIST listo y funcionando","version":"1.0","clases":["0"..."9"]}
```

### 4.9. Frontend — Canvas para Dibujar

Se creó el archivo frontend/index.html con las siguientes funcionalidades:

- Canvas HTML5 de 360x360 píxeles con fondo negro y trazo blanco, igual que el formato MNIST.
- Control de grosor del pincel con slider de 8 a 40 píxeles.
- Predicción automática: 800ms después de soltar el lápiz llama a la API sin necesidad de presionar botón.
- Panel de resultado: muestra el dígito detectado en grande con porcentaje de confianza.
- Barras de probabilidad: visualización de la confianza del modelo para los 10 dígitos simultáneamente.
- Historial de predicciones: guarda las últimas 12 predicciones de la sesión.
- Verificación de conexión: al cargar comprueba si la API está disponible y muestra el estado.

El frontend convierte el canvas a imagen PNG y la envía como multipart/form-data:

```bash
const blob = await new Promise(res => canvas.toBlob(res, 'image/png'));

const form = new FormData();

form.append('file', blob, 'digit.png');

const r = await fetch('http://localhost:8000/predict', { method: 'POST', body: form });
```

---

## 5. Problemas Encontrados y Soluciones

| **Problema** | **Causa** | **Solución** |
|----|----|----|
| python3.12 not found | Python 3.12 no estaba instalado | pyenv install 3.12.13 tras actualizar pyenv con brew |
| pyenv no cambia versión | Anaconda sobreescribe el PATH | eval "$(pyenv init -)" antes de usar pyenv |
| tensorflow-cpu no encontrado | No existe para ARM (Mac M3) | Usar tensorflow==2.20.0 en requirements.txt local |
| Docker build falla en FROM | Plataforma no especificada en M3 | Agregar --platform linux/amd64 en FROM del Dockerfile |
| Predicciones incorrectas | Canvas 360x360 vs MNIST 28x28 | Preprocesamiento: bounding box + cuadrado + LANCZOS |
| CORS bloqueado en navegador | FastAPI no tenía CORS configurado | Agregar CORSMiddleware en main.py |

---

## 6. Resultado Final

| **Componente** | **Estado** | **Detalle** |
|----|----|----|
| Python 3.12.13 |  Instalado | Via pyenv, coincide con Google Colab |
| Entorno virtual venv |  Activo | Dependencias aisladas del sistema |
| tensorflow 2.20.0 |  Instalado | Compatible con Apple Silicon M3 |
| API FastAPI local |  Probada | uvicorn main:app --reload |
| Imagen Docker |  Construida | img-api-mnist con --platform linux/amd64 |
| Contenedor Docker |  Corriendo | srv-api-mnist en puerto 8000 |
| Modelo Keras cargado |  Funcionando | modelo_mnist.keras desde api/model/ |
| Frontend Canvas |  Conectado | Dibuja y predice en tiempo real |
| Preprocesamiento |  Mejorado | Bounding box + LANCZOS + umbralización |
| Swagger UI |  Disponible | http://localhost:8000/docs |

---

## 7. Arquitectura del Sistema

## Flujo completo de una predicción:

1.  El usuario dibuja un dígito en el canvas HTML5 del navegador.

2.  Al soltar el lápiz, el JavaScript convierte el canvas a imagen PNG (800ms de debounce).

3.  La imagen se envía como multipart/form-data via fetch POST a http://localhost:8000/predict.

4.  Docker recibe la petición en el puerto 8000 y la redirige al contenedor srv-api-mnist.

5.  FastAPI (Uvicorn) procesa la petición en el endpoint /predict.

6.  El preprocesamiento normaliza la imagen: umbraliza, recorta, centra y redimensiona a 28x28.

7.  El modelo Keras realiza la inferencia y devuelve las probabilidades para los 10 dígitos.

8.  FastAPI responde con JSON: clase predicha, probabilidad y distribución completa.

9.  El frontend muestra el resultado, actualiza las barras y agrega al historial.

## Diagrama de componentes:

```bash
[ Navegador / Frontend HTML ]

| fetch POST /predict (imagen PNG)

↓

[ Docker Puerto 8000:8000 ]

|

↓

[ FastAPI + Uvicorn (main.py) ]

| preprocesamiento de imagen

↓

[ Modelo Keras (modelo_mnist.keras) ]

| predicción: 10 probabilidades

↓

[ JSON Response → Frontend ]
```

---

## 8. Referencia de Comandos Clave

| **Propósito** | **Comando** |
|----|----|
| Actualizar pyenv | brew update && brew upgrade pyenv |
| Instalar Python 3.12 | pyenv install 3.12.13 |
| Activar versión pyenv | eval "$(pyenv init -)" && pyenv local 3.12.13 |
| Crear entorno virtual | python -m venv venv |
| Activar entorno virtual | source venv/bin/activate |
| Instalar dependencias | pip install -r api/requirements.txt |
| Verificar TensorFlow | python -c "import tensorflow as tf; print(tf.__version__)" |
| Probar API local | uvicorn main:app --reload |
| Build Docker M3 | docker buildx build --platform linux/amd64 -t img-api-mnist . |
| Correr contenedor | docker run -d --name srv-api-mnist -p 8000:8000 img-api-mnist:latest |
| Ver contenedores activos | docker ps |
| Detener contenedor | docker stop srv-api-mnist |
| Eliminar contenedor | docker rm srv-api-mnist |
| Ver logs del contenedor | docker logs srv-api-mnist |
