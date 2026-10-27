# Hybrid U-Net para predicción de campos aerodinámicos

Implementación en Python de una arquitectura neuronal híbrida basada en redes convolucionales y U-Net para estimar campos aerodinámicos a partir de la geometría de un perfil.

El proyecto forma parte de una línea de investigación sobre modelos sustitutivos (*surrogate models*) de simulaciones de dinámica de fluidos computacional (CFD). Su objetivo es estudiar la predicción de campos de flujo mediante Deep Learning a partir de datos obtenidos con simulaciones numéricas.

## Arquitectura

La red recibe una función de distancia con signo (SDF) discretizada en una malla de **80 × 200** puntos y genera tres campos de salida:

- **Ux:** componente horizontal de la velocidad.
- **Uy:** componente vertical de la velocidad.
- **P:** campo de presión.

El modelo definido en `src/models/unet_hybrid.py` utiliza un codificador convolucional compartido, una etapa intermedia con capas densas y tres decodificadores independientes con conexiones de salto (*skip connections*). Cada decodificador reconstruye uno de los campos de salida.

El entrenamiento utiliza una función de pérdida MSE ponderada espacialmente mediante la SDF, con mayor peso relativo en regiones próximas a la geometría. El parámetro de ponderación se configura en `configs/config.yaml`.

## Tecnologías

- Python
- TensorFlow y Keras
- NumPy, Pandas y Scikit-learn
- Matplotlib
- PyYAML
- SciPy (interpolación durante el preprocesamiento)

## Organización del repositorio

~~~text
configs/config.yaml             Configuración del modelo, entrenamiento y evaluación
src/models/unet_hybrid.py       Definición de la arquitectura
src/preprocessing/build_dataset.py  Procesamiento de resultados CFD y construcción del dataset
src/training/losses.py         Funciones de pérdida y ponderación espacial
src/training/train.py          Entrenamiento y guardado de pesos
src/inference/predict.py       Carga del modelo y predicción de campos
src/inference/evaluate.py      Evaluación sobre el conjunto de test
src/visualization/plot_fields.py  Visualización y comparación de resultados
data/                          Datos de entrada y resultados
saved_models/                  Pesos, índices de partición y parámetros de normalización
~~~

## Instalación

Clonar el repositorio e instalar las dependencias en un entorno virtual de Python compatible con TensorFlow 2.16 o superior:

~~~bash
git clone https://github.com/nandoherrera97-code/Hybrid-UNET.git
cd Hybrid-UNET
python -m venv .venv
~~~

Activar el entorno virtual según el sistema operativo y ejecutar:

~~~bash
python -m pip install -r requirements.txt
python -m pip install scipy
~~~

**Nota:** SciPy se utiliza en `build_dataset.py`, aunque todavía no aparece en `requirements.txt`.

## Datos de entrada

El preprocesamiento espera encontrar los resultados de las simulaciones CFD y las matrices SDF con la siguiente estructura:

~~~text
data/raw/
├── sdf_df.pkl
└── results/
    └── <simulacion>/
        └── postProcessing/
            └── resultados.csv
~~~

Cada CSV debe contener las columnas `Points:0`, `Points:1`, `U:0`, `U:1` y `p`. El archivo `sdf_df.pkl` debe contener las columnas `Simulation` y `SDF`, usando los mismos identificadores de simulación.

El script interpola los campos CFD sobre una malla regular de **80 × 200** puntos y construye un DataFrame con las columnas `SDF`, `Ux_discretized`, `Uy_discretized` y `P_discretized`.

## Ejecución

### 1. Construcción del dataset

Con los datos de entrada disponibles:

~~~bash
python -m src.preprocessing.build_dataset
~~~

El script genera `data/raw/CNN_df.pkl` y guarda imágenes de los campos procesados en `data/raw/sdf/` y `data/raw/CFD_graphs/`. **Estas dos carpetas de imágenes se eliminan y regeneran al ejecutar el script.**

Actualmente, `configs/config.yaml` espera el dataset en `data/processed/CNN_df.pkl`. Por tanto, antes de entrenar es necesario mover o copiar el archivo generado a esa ruta, o ajustar la ruta de configuración.

### 2. Entrenamiento

~~~bash
python -m src.training.train
~~~

El entrenamiento separa los datos en conjuntos de entrenamiento (70 %), validación (15 %) y test (15 %). Los parámetros de normalización se calculan utilizando exclusivamente el conjunto de entrenamiento. Se emplea el optimizador Adam y una parada temprana basada en la pérdida de validación.

Los archivos generados se guardan en `saved_models/`, incluidos los pesos del mejor modelo, los índices de partición y los parámetros de normalización.

### 3. Evaluación

~~~bash
python -m src.inference.evaluate
~~~

La evaluación carga los pesos entrenados y utiliza el conjunto de test reservado. Calcula errores medios absolutos globales y cercanos a la pared (*near-wall*) para Ux, Uy y P, y genera comparaciones gráficas entre los campos de referencia y los predichos en `data/results/`.

Los errores porcentuales se normalizan mediante el valor medio absoluto del campo de referencia en el dominio fluido. La región *near-wall* se delimita con el umbral configurado en `evaluation.nearwall_cells`.

## Estado del proyecto

El repositorio incluye el código de la arquitectura, el preprocesamiento, el entrenamiento y la evaluación. **No incluye los datos CFD originales ni los pesos finales del modelo**, por lo que para ejecutar el flujo completo es necesario proporcionar los datos de entrada y entrenar el modelo.

El código se presenta como parte del trabajo de investigación y no como un sustituto validado de un solver CFD para aplicaciones de producción.

## Autor

Fernando Herrera Marín  
Investigación doctoral sobre Inteligencia Artificial aplicada a la Dinámica de Fluidos Computacional.
