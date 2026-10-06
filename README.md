# IA-_Predicción de Carbono Orgánico en Suelo (SOC)
# Predicción de Carbono Orgánico en el Suelo (SOC) en el Perú mediante IA

## 1. Descripción del Problema

El Carbono Orgánico del Suelo (SOC) es un indicador de las propiedades y condiciones del suelo, además de estar relacionado con su capacidad de almacenamiento de carbono. En el Perú, la disponibilidad de información edáfica detallada es limitada frente a la extensión y diversidad de los suelos del territorio nacional.

Los análisis tradicionales de laboratorio permiten obtener mediciones precisas de las propiedades del suelo, pero requieren recursos, tiempo y muestreo en campo. En este contexto, el uso de técnicas de aprendizaje automático puede contribuir a estimar el contenido de carbono orgánico a partir de otras propiedades edáficas disponibles.

Este proyecto utiliza la base de datos nacional AllpaDB para estudiar la relación entre las propiedades del suelo y el contenido de Carbono Orgánico del Suelo (`ORGC`), y desarrollar modelos predictivos mediante técnicas de Machine Learning.

## 2. Objetivo General y Específicos

* **Objetivo General:** Desarrollar y evaluar modelos de aprendizaje automático para predecir el contenido de Carbono Orgánico del Suelo (`ORGC`) utilizando información edáfica de la base de datos AllpaDB.

* **Objetivos Específicos:**
  1. Realizar un análisis exploratorio de datos (EDA) para caracterizar las variables edáficas y estudiar su relación con el contenido de carbono orgánico (`notebooks/01_eda.ipynb`).
  2. Implementar un modelo de Regresión Lineal Múltiple como baseline para establecer una referencia de rendimiento (`notebooks/02_modelos.ipynb`).
  3. Implementar y comparar modelos de aprendizaje automático, incluyendo Random Forest y XGBoost, y evaluar posteriormente el uso de una red neuronal MLP.
  4. Comparar el rendimiento de los modelos mediante métricas de regresión como $R^2$, RMSE y MAE.

## 3. Metodología

La metodología toma como referencia trabajos relacionados con el mapeo digital de suelos y la predicción del carbono orgánico mediante aprendizaje automático, entre ellos Padarian et al. (2019) y Salazar-Coronel et al. (2026).

1. **Dataset:** Se utilizará la base de datos AllpaDB, específicamente el archivo `allpaDB_layer_HARMONIZED_v1.csv`, que contiene 24,341 registros correspondientes a diferentes capas de perfiles de suelo.

2. **Variable objetivo:** Se utilizará `ORGC` como variable objetivo para la predicción del contenido de carbono orgánico del suelo.

3. **Análisis exploratorio:** Se analizarán las distribuciones de las variables, valores faltantes, valores atípicos, relaciones entre variables y correlaciones con la variable objetivo (`notebooks/01_eda.ipynb`).

4. **Preprocesamiento:** Se implementará un pipeline de preprocesamiento para el tratamiento de valores faltantes y, cuando corresponda según el modelo, el escalamiento de las variables predictoras. El preprocesamiento será ajustado utilizando únicamente los datos de entrenamiento para evitar fuga de información.

5. **Modelamiento:** Se establecerá una Regresión Lineal Múltiple como baseline y posteriormente se evaluarán modelos basados en árboles, como Random Forest y XGBoost, así como una red neuronal MLP.

6. **Evaluación:** Los modelos serán comparados mediante métricas de regresión como $R^2$, RMSE y MAE.

## 4. Revisión de Literatura (`papers/`)

* **Paper 1 (Nacional):** Salazar-Coronel et al. (2026). *Soil organic carbon content mapping along the coast of northern Peru: an ensemble machine learning approach*. Frontiers in Soil Science.

* **Paper 2 (Metodológico/Deep Learning):** Padarian, J., Minasny, B., & McBratney, A. B. (2019). *Using deep learning for digital soil mapping*. SOIL, 5(1), 79-89.

## 5. Integrantes del Grupo

* Israel Cristobal Cortavarria - 20182712
* Susan Sayli Lozano Bernardo - 20215871
* Juan Pablo Huamán Chara - 20227179
* Jean Marcos Huaroto Romero - 20193363
