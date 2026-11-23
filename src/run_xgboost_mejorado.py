"""Mejora de XGBoost usando el preprocesamiento actualizado (data/processed/).

Uso (desde la raiz del repositorio):
    python src/run_xgboost_mejorado.py

El preprocesamiento actualizado (notebooks/02_preprocesamiento.ipynb) cambia las variables
(depth_mid, thickness, sin SAND), aplica log1p y clipping a features asimetricas, imputa
dentro de train y usa otra division 80/20 por perfil. Por eso se reevalua tambien el
XGBoost original y Random Forest SOBRE LOS MISMOS DATOS: es la unica comparacion justa.

Genera results/xgboost/metrics_xgboost.csv, xgboost_mejorado.txt y plots/.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from evaluation import METRIC_NAMES, evaluate_models, cross_validate_grouped
from models import (create_random_forest, create_xgboost, create_xgboost_log,
                    tune_xgboost_log)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'xgboost'
SEED = 42


def load():
    X_train = pd.read_csv(PROC / 'X_train.csv')
    X_test = pd.read_csv(PROC / 'X_test.csv')
    y_train = pd.read_csv(PROC / 'y_train.csv')['ORGC']
    y_test = pd.read_csv(PROC / 'y_test.csv')['ORGC']
    g_train = pd.read_csv(PROC / 'groups_train.csv')['profile_code']
    return X_train, X_test, y_train, y_test, g_train


def main():
    (OUT / 'plots').mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, g_train = load()
    print(f'train {X_train.shape}, test {X_test.shape}, perfiles train {g_train.nunique()}')

    print('Ajustando hiperparametros de XGBoost (target log, CV agrupada)...')
    search = tune_xgboost_log(X_train, y_train, g_train, random_state=SEED)
    best_params = {k.replace('regressor__model__', ''): v for k, v in search.best_params_.items()}
    print('Mejores hiperparametros:', best_params, '| MAE CV:', round(-search.best_score_, 4))

    models = {
        'Random Forest (referencia)': create_random_forest(),
        'XGBoost original': create_xgboost(),
        'XGBoost + target log': create_xgboost_log(),
        'XGBoost mejorado (log + ajuste)': search.best_estimator_,
    }
    results = evaluate_models(models, X_train, y_train, X_test, y_test)
    _, cv = cross_validate_grouped(models, X_train, y_train, g_train, n_splits=5)

    test = results[results['Conjunto'] == 'test'].set_index('Modelo')[METRIC_NAMES].add_suffix('_test')
    table = test.loc[list(models)]
    table['R2_train'] = results[results['Conjunto'] == 'train'].set_index('Modelo')['R2']
    for m in ['R2', 'RMSE', 'MAE']:
        table[f'{m}_cv_mean'] = cv[(m, 'mean')]
        table[f'{m}_cv_std'] = cv[(m, 'std')]
    table = table.round(4).reset_index()
    table.to_csv(OUT / 'metrics_xgboost.csv', index=False)

    best = clone_fit(search.best_estimator_, X_train, y_train)
    pred = best.predict(X_test)
    rangos = pd.cut(y_test, [-np.inf, 1, 3, 10, np.inf], labels=['< 1 %', '1 - 3 %', '3 - 10 %', '> 10 %'])
    err = pd.DataFrame({'n': y_test.groupby(rangos, observed=False).size(),
                        'MAE': np.abs(y_test - pred).groupby(rangos, observed=False).mean().round(3)})
    perm = permutation_importance(best, X_test, y_test, scoring='neg_mean_absolute_error',
                                  n_repeats=5, random_state=SEED)
    imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False).round(3)

    with open(OUT / 'xgboost_mejorado.txt', 'w', encoding='utf-8') as f:
        f.write('XGBoost mejorado con el preprocesamiento actualizado\n' + '=' * 55 + '\n')
        f.write(f'Division por perfil: {len(X_train)} filas de entrenamiento, {len(X_test)} de prueba.\n')
        f.write(f'Hiperparametros (CV agrupada 3 folds, MAE): {best_params}\n\n')
        f.write('METRICAS (escala original de ORGC, %)\n' + table.set_index('Modelo').to_string() + '\n\n')
        f.write('MAE POR RANGO DE ORGC REAL (prueba)\n' + err.to_string() + '\n\n')
        f.write('IMPORTANCIA POR PERMUTACION (aumento del MAE)\n' + imp.to_string() + '\n')
    print(open(OUT / 'xgboost_mejorado.txt', encoding='utf-8').read())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_test, pred, s=6, alpha=0.4, color='#e76f51')
    lim = [0, max(y_test.max(), pred.max())]
    axes[0].plot(lim, lim, 'k--', lw=1)
    axes[0].set(xlabel='ORGC real (%)', ylabel='ORGC predicho (%)', title='XGBoost mejorado: real vs predicho')
    imp.sort_values().plot.barh(ax=axes[1], color='#e76f51')
    axes[1].set(title='Importancia por permutacion (aumento del MAE)')
    plt.tight_layout()
    plt.savefig(OUT / 'plots' / 'xgboost_mejorado.png', dpi=130)


def clone_fit(model, X, y):
    from sklearn.base import clone
    return clone(model).fit(X, y)


if __name__ == '__main__':
    main()
