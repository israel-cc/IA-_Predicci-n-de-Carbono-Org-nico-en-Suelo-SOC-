# Reporte de Estado - Entregable Parcial

**Curso:** Inteligencia Artificial Aplicada (1INF62) - PUCP  
**Proyecto:** Predicción de Carbono Orgánico en Suelos del Perú (AllpaDB)

---

## Checklist de Requisitos Exigidos

- [x] **1. Estructura del Repositorio:** Repositorio GitHub configurado con las carpetas `data/`, `notebooks/`, `papers/`, `src/` y `results/`.

- [x] **2. Documentación Base:** Archivo `README.md` con descripción del problema, objetivos, metodología e integrantes.

- [x] **3. Revisión de Literatura (`papers/`):** Artículos científicos relacionados con la predicción y mapeo del carbono orgánico del suelo mediante aprendizaje automático.

- [x] **4. Dataset Seleccionado (`data/raw/`):** Base de datos AllpaDB cargada en el repositorio, incluyendo el archivo `allpaDB_layer_HARMONIZED_v1.csv` y su diccionario de variables.

- [x] **5. Análisis Exploratorio de Datos (`notebooks/01_eda.ipynb`):** Análisis de estadísticas descriptivas, valores faltantes, distribuciones, correlaciones, valores atípicos y consistencia de variables.

- [x] **6. Preprocesamiento Inicial (`src/preprocessing.py`):** Implementación de funciones para separar variables predictoras y variable objetivo, imputar valores faltantes y preparar el escalamiento de las variables.

- [ ] **7. Modelo Baseline (`notebooks/02_modelos.ipynb`):** Implementación y evaluación de un modelo de Regresión Lineal Múltiple.

- [ ] **8. Modelos Avanzados:** Implementación y comparación de Random Forest, XGBoost y MLP.

---

## Estado Actual

Hasta el momento se ha completado la preparación inicial del proyecto, incluyendo la organización del repositorio, incorporación del dataset y artículos científicos, análisis exploratorio de los datos y desarrollo inicial del módulo de preprocesamiento.

El dataset contiene 24,341 registros correspondientes a diferentes capas de perfiles de suelo. La variable objetivo seleccionada es `ORGC`, correspondiente al contenido de carbono orgánico del suelo.

El siguiente paso es implementar el pipeline de entrenamiento y evaluación del modelo baseline, asegurando que el preprocesamiento se ajuste únicamente con los datos de entrenamiento para evitar fuga de información.

## Próximas Actividades

1. Definir la división de entrenamiento y prueba.
2. Integrar el preprocesamiento mediante `Pipeline` de scikit-learn.
3. Entrenar y evaluar la Regresión Lineal Múltiple.
4. Establecer las métricas baseline.
5. Implementar y comparar modelos de Machine Learning más avanzados.
