import numpy as np

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from preprocessing import create_preprocessor_scaled, create_preprocessor_unscaled


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



# Random Forest: modelo no lineal basado en arboles, el mas recomendado por la literatura
# revisada (Salazar-Coronel 2026; Carbajal 2024). Solo se imputa (mediana), sin escalar,
# porque los arboles no dependen de la escala. Se entrena sobre ORGC sin transformar.
def create_random_forest(n_estimators=300, max_depth=None, min_samples_leaf=5,
                         max_features=1.0, random_state=42):
    return Pipeline([
        ('preprocessor', create_preprocessor_unscaled()),
        ('model', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=random_state,
        ))
    ])


# Rejilla de hiperparametros de Random Forest (prefijo model__ = paso del Pipeline)
RF_PARAM_GRID = {
    'model__max_depth': [10, 20, None],
    'model__min_samples_leaf': [1, 5, 10],
    'model__max_features': [0.5, 1.0],
}


# Busca los mejores hiperparametros SOLO con el conjunto de entrenamiento. Usa validacion
# cruzada agrupada por perfil (3 folds) para que el ajuste no vea capas del mismo perfil
# en entrenamiento y validacion. Se optimiza el MAE porque ORGC es muy asimetrica.
def _tune(estimator, param_grid, X_train, y_train, groups_train, n_splits=3):
    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=1,  # el paralelismo ya se usa dentro de cada modelo (n_jobs=-1)
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search


def tune_random_forest(X_train, y_train, groups_train, n_splits=3, random_state=42):
    return _tune(create_random_forest(random_state=random_state), RF_PARAM_GRID,
                 X_train, y_train, groups_train, n_splits)


# XGBoost: gradient boosting sobre arboles, el mejor modelo del paper de la costa norte de
# Peru (Salazar-Coronel 2026, R2 = 0.83). Igual que Random Forest: solo imputacion (XGBoost
# acepta NaN, pero se imputa para que ambos modelos reciban los mismos datos) y ORGC sin transformar.
def create_xgboost(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=5,
                   subsample=0.8, colsample_bytree=0.8, random_state=42):
    return Pipeline([
        ('preprocessor', create_preprocessor_unscaled()),
        ('model', XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=-1,
            random_state=random_state,
        ))
    ])


XGB_PARAM_GRID = {
    'model__max_depth': [3, 6, 9],
    'model__learning_rate': [0.03, 0.1],
    'model__min_child_weight': [1, 5, 10],
}


def tune_xgboost(X_train, y_train, groups_train, n_splits=3, random_state=42):
    return _tune(create_xgboost(random_state=random_state), XGB_PARAM_GRID,
                 X_train, y_train, groups_train, n_splits)


# Diccionario con todos los modelos de la comparacion: baselines + Random Forest + XGBoost
def get_all_models(random_forest=None, xgboost=None):
    models = get_baseline_models()
    models['Random Forest'] = random_forest if random_forest is not None else create_random_forest()
    models['XGBoost'] = xgboost if xgboost is not None else create_xgboost()
    return models
