# Reporte de Estado - Entregable Final

**Curso:** Inteligencia Artificial Aplicada (1INF62) - PUCP  
**Proyecto:** Predicción de Carbono Orgánico en Suelos del Perú (AllpaDB)

---

## Checklist de Requisitos Exigidos

- [x] **1. Código limpio, comentado y reproducible:** `src/preprocessing.py`, `src/models.py`, `src/evaluation.py`, `src/run_comparison.py`, `src/run_xgboost_mejorado.py`, `src/run_xgboost_28vars.py`, `src/run_all_28vars.py` y `src/run_cv_28vars.py`. Se reproduce con `python src/run_comparison.py` (comparación inicial), `python src/run_xgboost_mejorado.py` (XGBoost con el preprocesamiento actualizado) `python src/run_xgboost_28vars.py` (XGBoost con 28 variables), `python src/run_all_28vars.py` (comparación final con 28 variables) y `python src/run_cv_28vars.py` (validación cruzada) desde la raíz del repositorio, o ejecutando `notebooks/02_modelos.ipynb`. Semilla fija (`random_state=42`).

- [x] **2. `requirements.txt`:** dependencias necesarias (numpy, pandas, scikit-learn, xgboost, matplotlib y Jupyter).

- [x] **3. Modelos implementados en `src/models.py`:** baselines (Media, Mediana, Regresión Lineal y Regresión Lineal sobre `log1p(ORGC)`), **Random Forest** y **XGBoost**, ambos con ajuste de hiperparámetros solo sobre entrenamiento (`GridSearchCV` + `GroupKFold`). Para el preprocesamiento actualizado se agregó `create_xgboost_log` / `tune_xgboost_log` (XGBoost entrenado sobre `log1p(ORGC)`). Para el experimento final con 28 variables están `create_mlp_28`, `create_cubist_28`, `create_random_forest_28` y `create_xgboost_28` (MLP, Cubist, Random Forest y XGBoost).

- [x] **4. Tabla comparativa y gráficos:** [`results/metrics.csv`](results/metrics.csv) (todos los modelos con 8 y con 28 variables) y [`results/plots/`](results/plots/) (comparación de métricas, real vs. predicho, MAE por rango de ORGC e importancia de variables). Resultados del XGBoost mejorado en [`results/xgboost/`](results/xgboost/) del XGBoost con 28 variables en [`results/xgboost_28vars/`](results/xgboost_28vars/) y de la comparación final con 28 variables en [`results/final_28vars/`](results/final_28vars/).

- [x] **5. `results/model_comparison.txt`:** resumen de resultados con 8 y 28 variables, validación cruzada, mejor modelo, insights y limitaciones.

- [x] **6. Análisis de por qué un modelo funcionó mejor, limitaciones e insights:** secciones V y VI de `report.pdf` (todos los modelos con 28 variables) y `results/model_comparison.txt`.

- [x] **7. Paper/Reporte académico (3-5 páginas) en `report.pdf`:** 4 páginas, generado desde `report/report.html`. Todos los modelos comparados con 28 variables (`python src/run_all_28vars.py`, resultados en [`results/final_28vars/`](results/final_28vars/)).

- [ ] **8. Presentación (máximo 5 minutos) en `presentation.pdf`:** a cargo de otro integrante.

---

## Métricas Utilizadas

`ORGC` es una concentración (%) muy asimétrica: la mayoría de los suelos tiene valores bajos y pocos superan 10 %. Por eso se usan varias métricas, todas en la escala original de `ORGC`:

| Métrica | Por qué |
|---|---|
| **MAE** | Error promedio en puntos porcentuales de ORGC; fácil de interpretar y poco dominado por valores extremos. |
| **MedAE** | Error de la predicción típica; refleja el desempeño en la gran mayoría de suelos. |
| **RMSE** | Penaliza las fallas grandes, que ocurren en suelos ricos en carbono. |
| **R²** | Fracción de varianza explicada; permite comparar con los papers revisados. |

---

## Resultados con el preprocesamiento inicial (`src/preprocessing.py`)

Conjunto de prueba, división 80/20 agrupada por perfil. Estos números **no son comparables** con los de la sección siguiente: cambian las variables y la partición.

| Modelo | R² | RMSE | MAE | MedAE |
|---|---|---|---|---|
| Media | -0.000 | 5.10 | 2.16 | 1.50 |
| Mediana | -0.061 | 5.26 | 1.64 | 0.49 |
| Regresión Lineal | 0.093 | 4.86 | 2.12 | 1.28 |
| Regresión Lineal (log ORGC) | 0.053 | 4.97 | 1.54 | 0.52 |
| **Random Forest** | **0.281** | **4.33** | 1.63 | 0.56 |
| XGBoost | 0.233 | 4.47 | 1.66 | 0.53 |

Random Forest y XGBoost superan a la regresión lineal en R² y RMSE. Entre ellos están empatados en validación cruzada agrupada (R² = 0.285 ± 0.041 frente a 0.282 ± 0.037); Random Forest queda algo mejor en la división única y XGBoost sobreajusta más. Ninguno gana en MAE ni MedAE, donde la mediana y la regresión lineal sobre log(ORGC) son iguales o mejores. Todos los modelos fallan en suelos con ORGC > 10 %. Detalle en `results/model_comparison.txt`.

---

## XGBoost con el preprocesamiento actualizado (8 variables)

El preprocesamiento actualizado (`notebooks/02_preprocesamiento.ipynb`, datos en `data/processed/`) reemplaza `upper_depth`/`lower_depth` por `depth_mid` y `thickness`, elimina `SAND` (redundante con `SILT` y `CLAY`), aplica `log1p` y recorte por IQR a las variables asimétricas, imputa solo con estadísticos de entrenamiento y usa otra división 80/20 por perfil (17 338 filas de entrenamiento y 4 263 de prueba). Siguiendo la recomendación de ese notebook, el modelo final se entrena sobre `ORGC_log = log1p(ORGC)` y sus predicciones se devuelven a % con `expm1` antes de calcular las métricas, así que todas están en la escala original de `ORGC`.

**Hiperparámetros** (validación cruzada agrupada por perfil de 3 folds, minimizando el MAE): `max_depth=7`, `learning_rate=0.03`, `n_estimators=600`, `min_child_weight=1`, `reg_lambda=10`.

Comparación sobre los **mismos datos** (conjunto de prueba):

| Modelo | R² | RMSE | MAE | MedAE | R² entrenamiento |
|---|---|---|---|---|---|
| Random Forest (referencia) | 0.271 | 4.69 | 1.63 | 0.54 | 0.565 |
| XGBoost original (ORGC sin transformar) | 0.273 | 4.69 | 1.67 | 0.58 | 0.627 |
| XGBoost + target log | 0.232 | 4.82 | 1.35 | 0.37 | 0.408 |
| **XGBoost final (log + ajuste)** | 0.242 | 4.78 | **1.33** | **0.35** | 0.472 |

**Por qué se elige el XGBoost final**
- Es el que sigue la indicación del preprocesamiento (entrenar sobre `ORGC_log`).
- Tiene el menor MAE (1.33) y MedAE (0.35): reduce el MAE un 20 % frente al XGBoost sin transformar y acierta más en las capas con ORGC < 1 %, que son el 63 % de la prueba.
- Se sobreajusta menos (R² de entrenamiento 0.47 frente a 0.63).

**Limitaciones**
- No supera a Random Forest ni al XGBoost sin transformar en R² ni RMSE (0.24 frente a 0.27): entrenar sobre el logaritmo mejora el error típico a costa de los suelos ricos en carbono.
- Sigue fallando en suelos con ORGC > 10 % (MAE ≈ 19.9 puntos en 137 capas); por rango de ORGC el MAE es 0.40 (< 1 %), 0.80 (1-3 %), 2.79 (3-10 %) y 19.9 (> 10 %).
- Variables más importantes (permutación, aumento del MAE): `PHAQ` (0.28), `depth_mid` (0.26) y `ECEC` (0.24).
- Detalle completo en [`results/xgboost/xgboost_mejorado.txt`](results/xgboost/xgboost_mejorado.txt).

---

## Modelo final: XGBoost con 28 variables

Con 8 variables, ajustar hiperparámetros apenas mejoraba los modelos: el límite era la información, no el algoritmo. Siguiendo el Random Forest ampliado (`notebooks/Random_forest_Huaroto.ipynb`), se agregaron a las 8 variables de `data/processed/` otras 20 que ya existen en AllpaDB y no miden el carbono directamente: **horizonte** (A, B, C, O, H, E, R u otro; one-hot), **cationes intercambiables** (`ECA`, `EMG`, `EK`, `ENA`), **acidez intercambiable** (`EAH`), **CIC a pH 7** (`CECPH7`), **color Munsell húmedo** (value y chroma), **posición de la capa** en el perfil (`rank`) y **fuente** del dato (one-hot). Los faltantes se imputan con la mediana de entrenamiento. Se excluyen `BD2` (densidad aparente estimada a partir del propio carbono: fuga de información) y `NITKJD` (se mide en el mismo análisis que el carbono). Se usa la misma división por perfil, así que el test es idéntico al de las secciones anteriores.

**Entrenamiento** (`src/run_xgboost_28vars.py`): objetivo `ORGC_log`; hiperparámetros elegidos en una validación interna (20 % de los perfiles de entrenamiento) con *early stopping*: `max_depth=10`, `learning_rate=0.03`, `min_child_weight=1`, `colsample_bytree=0.8`, `subsample=0.8`, `reg_lambda=10`, 1009 árboles. El modelo final se reentrena con todo el entrenamiento y se evalúa una sola vez en test.

Conjunto de prueba (mismos datos para los tres modelos):

| Modelo | R² (log) | R² (%) | RMSE (%) | MAE (%) | MedAE (%) | Error ≤ ±0.25 % | Error ≤ ±1 % |
|---|---|---|---|---|---|---|---|
| XGBoost, 8 variables | 0.466 | 0.250 | 4.76 | 1.34 | 0.34 | 40.7 % | 78.2 % |
| Random Forest, 28 variables | 0.651 | 0.423 | 4.18 | 1.09 | 0.29 | 45.6 % | **83.9 %** |
| **XGBoost, 28 variables** | **0.667** | **0.454** | **4.06** | **1.07** | **0.26** | **48.3 %** | 83.7 % |

**Por qué se elige**
- Agregar las 20 variables sube el R² en % de 0.25 a 0.45 y baja el MAE de 1.34 a 1.07 (−20 %): la mejora viene de la información, no del algoritmo.
- Con las mismas variables, XGBoost supera ligeramente al Random Forest en R², RMSE, MAE y MedAE, y es el único que mejora claramente en los suelos ricos en carbono (> 10 %: MAE 15.1 frente a 16.3).
- Las variables más importantes son el horizonte (A, O y H), la fuente del dato y la acidez intercambiable: el tipo de capa explica mucho del carbono.

**Limitaciones**
- La diferencia con el Random Forest de 28 variables es pequeña en prueba, pero la validación cruzada agrupada (5 folds) confirma a XGBoost como el mejor: R² log 0.68 ± 0.02 frente a 0.65 ± 0.02.
- Sobreajusta más (R² de entrenamiento 0.98 en escala log frente a 0.67 en test).
- Sigue fallando en suelos orgánicos (> 10 %, 137 capas): MAE ≈ 15 puntos. Por rango: 0.29 (< 1 %), 0.72 (1-3 %), 2.49 (3-10 %), 15.1 (> 10 %).
- Las nuevas variables tienen muchos faltantes (color Munsell 52 %, CIC pH 7 48 %) y la fuente del dato es importante, lo que puede reflejar diferencias de método entre laboratorios más que propiedades del suelo.
- Detalle completo en [`results/xgboost_28vars/xgboost_28vars.txt`](results/xgboost_28vars/xgboost_28vars.txt).

---

## Estado Actual

El modelo final es XGBoost con 28 variables, comparado contra todos los modelos del grupo (MLP, Cubist, Random Forest, regresión lineal y media) con las mismas 28 variables, la misma partición por perfil y validación cruzada agrupada. El reporte académico está en `report.pdf`. La presentación está a cargo de otro integrante.
