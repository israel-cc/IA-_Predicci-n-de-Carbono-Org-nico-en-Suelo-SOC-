# Reporte de Estado - Entregable Parcial

**Curso:** Inteligencia Artificial Aplicada (1INF62) - PUCP  
**Proyecto:** Predicción de Carbono Orgánico en Suelos del Perú (AllpaDB)

---

## Checklist de Requisitos Exigidos

- [x] **1. Estructura del Repositorio:** Repositorio GitHub con las carpetas `data/`, `notebooks/`, `papers/` y `src/`.

- [x] **2. Documentación Base:** `README.md` con descripción del problema, objetivos, metodología e integrantes.

- [x] **3. Revisión de Literatura (`papers/`):** Artículos científicos sobre predicción y mapeo del carbono orgánico del suelo mediante aprendizaje automático (Paper 1 a Paper 4, con resúmenes).

- [x] **4. Dataset Seleccionado (`data/raw/`):** Base de datos AllpaDB (`allpaDB_layer_HARMONIZED_v1.csv` y su diccionario). El origen, las características y la licencia (CC BY 4.0) están documentados en [`data/raw/README.md`](data/raw/README.md).

- [x] **5. Análisis Exploratorio de Datos (`notebooks/01_eda.ipynb`):** Estadísticas descriptivas, valores faltantes, distribuciones, correlaciones, valores atípicos y consistencia de variables, con visualizaciones.

- [x] **6. Preprocesamiento Inicial:** Funciones en `src/preprocessing.py` (separación de variables, imputación y escalamiento) y limpieza documentada en `notebooks/02_preprocesamiento.ipynb`.

- [x] **7. Modelo Baseline (`notebooks/02_modelos.ipynb`):** Implementados y evaluados la Media, la Mediana y la Regresión Lineal Múltiple (con y sin transformación `log1p` del target). Código en `src/models.py` y `src/evaluation.py`.

- [ ] **8. Propuesta de Modelos desde la Literatura:** Listado de técnicas identificadas en los papers para implementar (en elaboración por el grupo).

- [ ] **9. Modelos Avanzados:** Implementación y comparación de Random Forest, XGBoost y MLP.

---

## Resultados del Baseline

División entrenamiento/prueba 80/20 **agrupada por perfil** (`profile_code`), para que las capas de un mismo perfil no queden en ambos conjuntos. Métricas en conjunto de prueba, escala original de `ORGC` (%):

| Modelo | R² | RMSE | MAE |
|---|---|---|---|
| Media | -0.000 | 5.10 | 2.16 |
| Mediana | -0.061 | 5.26 | 1.64 |
| Regresión Lineal | 0.093 | 4.86 | 2.12 |
| Regresión Lineal (log ORGC) | 0.053 | 4.97 | 1.54 |

La validación cruzada agrupada de 5 folds da resultados consistentes (R² de la Regresión Lineal = 0.089).

**Hallazgos principales:**
- Las variables edáficas disponibles explican linealmente solo cerca del 9 % de la varianza de `ORGC`.
- Los modelos lineales subestiman los valores altos de `ORGC` y la Regresión Lineal produce un 10.4 % de predicciones negativas.
- Estos valores son la referencia que deben superar los modelos no lineales.

## Estado Actual

Se completaron la organización del repositorio, la incorporación y documentación del dataset, la revisión de literatura, el análisis exploratorio, el preprocesamiento y el baseline.

El dataset contiene 24,341 capas de suelo de 8,793 perfiles. La variable objetivo es `ORGC` (carbono orgánico por Walkley & Black, %). Tras eliminar filas sin `ORGC` y valores mayores a 100 %, quedan 21,601 filas.

## Próximas Actividades

1. Completar la propuesta de modelos a partir de la literatura.
2. Implementar y comparar Random Forest y XGBoost con la misma división agrupada y las mismas métricas.
3. Evaluar una red neuronal MLP.
4. Analizar la importancia de variables y comparar contra el baseline.
