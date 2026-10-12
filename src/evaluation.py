import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# ORGC esta en porcentaje: valores mayores a 100 son imposibles
MAX_ORGC = 100


# Elimina las filas con ORGC imposible (> 100 %)
def filter_valid_target(X, y, groups):
    valid = y <= MAX_ORGC
    return X[valid], y[valid], groups[valid]


# Division entrenamiento/prueba agrupada por perfil: todas las capas de un mismo
# perfil quedan en el mismo conjunto, para evitar fuga de informacion entre capas
def group_train_test_split(X, y, groups, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            groups.iloc[train_idx], groups.iloc[test_idx])


# Metricas de regresion en la escala original de ORGC
def regression_metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
    }


# Entrena cada modelo con el conjunto de entrenamiento y lo evalua en entrenamiento y prueba
def evaluate_models(models, X_train, y_train, X_test, y_test):
    rows = []
    for name, model in models.items():
        model = clone(model)
        model.fit(X_train, y_train)
        for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
            rows.append({'Modelo': name, 'Conjunto': split, **regression_metrics(y, model.predict(X))})
    return pd.DataFrame(rows)


# Validacion cruzada agrupada por perfil. Devuelve media y desviacion de cada metrica
def cross_validate_grouped(models, X, y, groups, n_splits=5):
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    rows = []
    for name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(folds):
            fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            metrics = regression_metrics(y.iloc[test_idx], fitted.predict(X.iloc[test_idx]))
            rows.append({'Modelo': name, 'Fold': fold, **metrics})
    results = pd.DataFrame(rows)
    summary = results.groupby('Modelo', sort=False)[['R2', 'RMSE', 'MAE']].agg(['mean', 'std'])
    return results, summary
