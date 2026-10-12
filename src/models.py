import numpy as np

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor_scaled


# Predictor de la media: ignora las variables predictoras y siempre predice
# la media de ORGC en entrenamiento. Es el piso de rendimiento (R2 ~ 0).
def create_mean_baseline():
    return DummyRegressor(strategy='mean')


# Predictor de la mediana: igual que el anterior, pero con la mediana. ORGC es muy
# asimetrica, por lo que la mediana minimiza el MAE y la media minimiza el RMSE.
def create_median_baseline():
    return DummyRegressor(strategy='median')


# Regresion Lineal Multiple con imputacion y escalamiento ajustados solo con entrenamiento
def create_linear_regression():
    return Pipeline([
        ('preprocessor', create_preprocessor_scaled()),
        ('model', LinearRegression())
    ])


# Regresion Lineal Multiple entrenada sobre log(1 + ORGC) para reducir la asimetria.
# Las predicciones se devuelven en la escala original, por lo que las metricas son comparables.
def create_linear_regression_log():
    return TransformedTargetRegressor(
        regressor=create_linear_regression(),
        func=np.log1p,
        inverse_func=np.expm1
    )


# Diccionario con todos los modelos baseline
def get_baseline_models():
    return {
        'Media': create_mean_baseline(),
        'Mediana': create_median_baseline(),
        'Regresion Lineal': create_linear_regression(),
        'Regresion Lineal (log ORGC)': create_linear_regression_log(),
    }
