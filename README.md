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

La metodología toma como referencia trabajos relacionados con el mapeo digital de suelos y la predicción del carbono orgánico mediante aprendizaje automático, entre ellos Salazar-Coronel et al. (2026), Carbajal et al. (2024, 2025), Ou et al. (2024) y Benke et al. (2020).

1. **Dataset:** Se utilizará la base de datos AllpaDB, específicamente el archivo `allpaDB_layer_HARMONIZED_v1.csv`, que contiene 24,341 registros correspondientes a diferentes capas de perfiles de suelo.

2. **Variable objetivo:** Se utilizará `ORGC` como variable objetivo para la predicción del contenido de carbono orgánico del suelo.

3. **Análisis exploratorio:** Se analizarán las distribuciones de las variables, valores faltantes, valores atípicos, relaciones entre variables y correlaciones con la variable objetivo (`notebooks/01_eda.ipynb`).

4. **Preprocesamiento:** Se implementará un pipeline de preprocesamiento para el tratamiento de valores faltantes y, cuando corresponda según el modelo, el escalamiento de las variables predictoras. El preprocesamiento será ajustado utilizando únicamente los datos de entrenamiento para evitar fuga de información.

5. **Modelamiento:** Se establecerá una Regresión Lineal Múltiple como baseline y posteriormente se evaluarán modelos basados en árboles, como Random Forest y XGBoost, así como Cubist y una red neuronal MLP.

6. **Evaluación:** Los modelos serán comparados mediante métricas de regresión como $R^2$, RMSE y MAE.

## 4. Revisión de Literatura (`papers/`)

Cada integrante revisó un paper; su resumen está en `papers/paper_integranteN.pdf` y los artículos originales disponibles están en la misma carpeta.

| Integrante | Paper revisado | Modelo tomado | Archivos |
|---|---|---|---|
| 1. Jean Marcos Huaroto Romero | Salazar-Coronel et al. (2026). *Soil organic carbon content mapping along the coast of northern Peru: an ensemble machine learning approach*. Front. Soil Sci. | Random Forest, XGBoost | `paper_integrante1.pdf`, `SalazarCoronel_2026.pdf` |
| 2. Israel Cristobal Cortavarria | Ou et al. (2024). *Improving soil organic carbon mapping in farmlands using machine learning models and complex cropping system information*. Environ. Sci. Eur. | Cubist | `paper_integrante2.pdf` |
| 3. Juan Pablo Huamán Chara | Carbajal et al. (2024). *From rangelands to cropland, land-use change and its impact on soil organic carbon variables in a Peruvian Andean highlands*. Ecosystems. | MLP | `paper_integrante3.pdf`, `Carbajal_2024.pdf` |
| 4. Susan Sayli Lozano Bernardo | Carbajal et al. (2025). *Spatial prediction of soil organic carbon stocks across contrasting Andean basins, Peru*. Geoderma Reg. | Boosting (XGBoost) | `paper_integrante4.pdf`, `Carbajal_2025.pdf` |

Referencia complementaria: Benke et al. (2020). *Development of pedotransfer functions by machine learning for prediction of soil electrical conductivity and organic carbon content*. Geoderma (`Benke_2020.pdf`).

## 5. Integrantes del Grupo

* Israel Cristobal Cortavarria - 20182712
* Susan Sayli Lozano Bernardo - 20215871
* Juan Pablo Huamán Chara - 20227179
* Jean Marcos Huaroto Romero - 20193363

## 6. Cómo ejecutar

Requiere Python 3.11+. Desde la raíz del repositorio:

```bash
pip install -r requirements.txt

python src/run_comparison.py     # comparación inicial con 8 variables (baselines, RF, XGBoost)
python src/run_mlp.py            # MLP con 8 variables
python src/run_xgboost_28vars.py # XGBoost con 28 variables (búsqueda de hiperparámetros)
python src/run_all_28vars.py     # comparación final de todos los modelos con 28 variables
python src/run_cv_28vars.py      # reajuste de Cubist/MLP y validación cruzada agrupada
```

Los notebooks `notebooks/01_eda.ipynb` (EDA), `notebooks/02_preprocesamiento.ipynb` y `notebooks/02_modelos.ipynb` documentan el análisis exploratorio, el preprocesamiento y los modelos. Los resultados se guardan en `results/` (tabla comparativa en `results/metrics.csv`, resumen en `results/model_comparison.txt`) y el reporte final está en `report.pdf`.

## 7. Resultados principales

| Modelo (28 variables) | R² | MAE (%) |
|---|---|---|
| Regresión lineal | 0.12 | 1.45 |
| MLP | 0.44 | 1.14 |
| Cubist | 0.40 | 1.10 |
| Random Forest | 0.42 | 1.09 |
| **XGBoost** | **0.45** | **1.07** |

XGBoost con 28 variables es el mejor modelo, también en validación cruzada agrupada por perfil.
