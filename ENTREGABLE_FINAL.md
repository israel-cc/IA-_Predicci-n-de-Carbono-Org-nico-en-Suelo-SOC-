# Reporte de Estado - Entregable Final

**Curso:** Inteligencia Artificial Aplicada (1INF62) - PUCP  
**Proyecto:** Predicción de Carbono Orgánico en Suelos del Perú (AllpaDB)

---

## Checklist de Requisitos Exigidos

- [x] **1. Código limpio, comentado y reproducible:** `src/preprocessing.py`, `src/models.py`, `src/evaluation.py` y `src/run_comparison.py`. Se reproduce con `python src/run_comparison.py` desde la raíz del repositorio o ejecutando `notebooks/02_modelos.ipynb`. Semilla fija (`random_state=42`).

- [x] **2. `requirements.txt`:** dependencias necesarias (numpy, pandas, scikit-learn, xgboost, matplotlib y Jupyter).

- [x] **3. Modelos implementados en `src/models.py`:** baselines (Media, Mediana, Regresión Lineal y Regresión Lineal sobre `log1p(ORGC)`), **Random Forest** y **XGBoost**, ambos con ajuste de hiperparámetros solo sobre entrenamiento (`GridSearchCV` + `GroupKFold`).

- [x] **4. Tabla comparativa y gráficos:** [`results/metrics.csv`](results/metrics.csv) y [`results/plots/`](results/plots/) (comparación de métricas, real vs. predicho, MAE por rango de ORGC e importancia de variables).

- [x] **5. `results/model_comparison.txt`:** resumen de resultados, justificación de las métricas, análisis detallado, limitaciones y siguientes pasos.

- [ ] **6. Análisis de por qué un modelo funcionó mejor, limitaciones e insights:** análisis detallado en `results/model_comparison.txt` (Random Forest y XGBoost vs. baseline). Falta actualizarlo si se incorpora MLP.

- [ ] **7. Paper/Reporte académico (3-5 páginas) en `report.pdf`:** pendiente.

- [ ] **8. Presentación (máximo 5 minutos) en `presentation.pdf`:** pendiente.

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

## Resultados (conjunto de prueba, división 80/20 agrupada por perfil)

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

## Estado Actual

La comparación incluye el baseline, Random Forest y XGBoost. Quedan pendientes el MLP (opcional), el reporte académico y la presentación.
