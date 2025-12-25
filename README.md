**Sistema Inteligente para la Clasificación de Riesgo Crediticio**

Breve guía para ejecutar el proyecto de ejemplo que combina Lógica Difusa y Machine Learning.

Requisitos:
- Python 3.9+
- Instalar dependencias: `pip install -r requirements.txt`

Uso rápido:
- Ejecutar con datos sintéticos (por defecto):
  `python -m src.main`
- Ejecutar con un CSV de Kaggle: (debe contener variables numéricas comunes y una columna `target`)
  `python -m src.main --data /ruta/dataset.csv`

# Sistema Inteligente para la Clasificación de Riesgo Crediticio

Breve guía para ejecutar el proyecto de ejemplo que combina Lógica Difusa y Machine Learning.

## Requisitos
- Python 3.9+
- Instalar dependencias: `pip install -r requirements.txt`

## Uso rápido
- Ejecutar con datos sintéticos (por defecto):
  ```powershell
  python -m src.main
  ```
- Ejecutar con un CSV de Kaggle: (debe contener variables numéricas comunes y una columna `target`)
  ```powershell
  python -m src.main --data /ruta/dataset.csv
  ```

## Salida
- `output/results.csv`: tabla con métricas por modelo
- `output/figures/`: gráficas de importancia / curvas

## Descripción breve de los módulos
- `src/fuzzy_module.py`: funciones de membresía y cálculo de riesgo difuso.
- `src/ml_module.py`: entrenamiento y evaluación de modelos ML.
- `src/utils.py`: carga de datos y generador sintético.
- `src/main.py`: script principal que orquesta la ejecución.

## Arquitectura en 3 capas (Descripción)

El proyecto está organizado conceptualmente en tres capas para facilitar el diagrama arquitectónico y el mantenimiento:

1️⃣ **Capa de Presentación**
- Interfaz web (Streamlit): `src/web_app.py` (helper disponible en `src/presentation/web_app.py`).
- Captura de datos desde formulario o CSV y visualización de resultados.

2️⃣ **Capa de Inteligencia**
- Módulo de lógica difusa: `src/fuzzy_module.py` (exportado en `src.intelligence`).
- Módulo de Machine Learning: `src/ml_module.py` (exportado en `src.intelligence`).
- Módulo de integración híbrida: `src/intelligence/integration.py` que coordina ambas salidas y elabora la decisión final.

3️⃣ **Capa de Datos**
- Dataset German Credit: `data/german_credit_data.csv` (accesible via `src.data.get_dataset_path()`).
- Modelos entrenados: `output/models/model_logistic.pkl` (ruta accesible via `src.data.get_model_path()`).
- Registros de evaluación y artefactos en `output/evaluations/`.

Los nuevos paquetes `src.presentation`, `src.intelligence` y `src.data` contienen wrappers y utilidades que facilitan la creación del diagrama en capas y mejoran la separación de responsabilidades.

## Notas
- Para ejecutar la app Streamlit desde la capa de presentación (PowerShell):
  ```powershell
  python -m streamlit run src/web_app.py
  ```

Licencia: sin licencia incluida.
