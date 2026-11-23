"""Entrena una red neuronal MLP para predecir ORGC y la compara con los demas modelos.

Uso (desde la raiz del repositorio, despues de run_comparison.py):
    python src/run_mlp.py

Genera:
    results/mlp/metrics_mlp.csv          metricas de la MLP
    results/mlp/hiperparametros_mlp.csv  resultados de la busqueda de hiperparametros
    results/mlp/mlp_comparison.txt       resumen numerico
    results/mlp/plots/*.png              graficos
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from evaluation import (METRIC_NAMES, cross_validate_grouped, evaluate_models,
                        group_train_test_split)
from preprocessing import FEATURES
from run_comparison import load_clean_data

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / 'results' / 'mlp'
PLOTS_DIR = RESULTS_DIR / 'plots'
SEED = 42

# Variables con cola larga (ver 02_preprocesamiento.ipynb). A la red le afectan mucho los
# valores extremos (ECEC llega a 2476), por eso se les aplica log1p antes de escalar
LOG_FEATURES = ['TCEQ', 'ELCOSP', 'ECEC']
OTHER_FEATURES = [f for f in FEATURES if f not in LOG_FEATURES]


# MLP: imputacion (mediana) + log1p en las variables asimetricas + escalamiento.
# La red necesita variables escaladas. Se entrena sobre log(1 + ORGC) y las predicciones
# se devuelven en %, igual que la Regresion Lineal (log ORGC)
def create_mlp(hidden_layer_sizes=(64, 32), alpha=0.001, learning_rate_init=0.001,
               random_state=SEED):
    preprocessor = ColumnTransformer([
        ('log', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(np.log1p)),
            ('scaler', StandardScaler())
        ]), LOG_FEATURES),
        ('otras', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), OTHER_FEATURES),
    ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=256,
            max_iter=500,
            early_stopping=True,  # usa el 10 % del entrenamiento para saber cuando parar
            n_iter_no_change=20,
            random_state=random_state,
        ))
    ])
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


# Rejilla de hiperparametros (prefijo regressor__model__ = paso de la MLP)
MLP_PARAM_GRID = {
    'regressor__model__hidden_layer_sizes': [(32,), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
    'regressor__model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 3.0],
    'regressor__model__learning_rate_init': [0.001, 0.003],
}


# Igual que con Random Forest y XGBoost: busqueda solo con entrenamiento, CV agrupada por
# perfil de 3 folds y se minimiza el MAE
def tune_mlp(X_train, y_train, groups_train):
    search = GridSearchCV(
        create_mlp(),
        param_grid=MLP_PARAM_GRID,
        scoring='neg_mean_absolute_error',
        cv=GroupKFold(n_splits=3),
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search


# MAE por rango de ORGC real (mismos rangos que results/model_comparison.txt)
def error_by_range(y_test, pred):
    bins = [0, 1, 3, 10, np.inf]
    labels = ['< 1 %', '1 - 3 %', '3 - 10 %', '> 10 %']
    ranges = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)
    abs_err = (y_test - pred).abs()
    return pd.DataFrame({
        'n': ranges.value_counts().reindex(labels),
        'MAE': abs_err.groupby(ranges, observed=False).mean().reindex(labels),
    })


def plot_learning_curve(mlp):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(mlp.loss_curve_)
    ax[0].set_title('Perdida de entrenamiento')
    ax[0].set_xlabel('Epoca')
    ax[1].plot(mlp.validation_scores_, color='#e76f51')
    ax[1].set_title('R2 de validacion (en log ORGC)')
    ax[1].set_xlabel('Epoca')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'curva_aprendizaje.png', dpi=100)
    plt.close(fig)


def plot_predictions(y_test, pred):
    lim = y_test.quantile(0.99)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, pred, s=6, alpha=0.3, color='#6a4c93')
    ax.plot([0, lim], [0, lim], 'r--', label='Prediccion perfecta')
    ax.set_xlim(0, lim)
    ax.set_ylim(-1, lim)
    ax.set_xlabel('ORGC real (%)')
    ax.set_ylabel('ORGC predicho (%)')
    ax.set_title('MLP')
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'real_vs_predicho_mlp.png', dpi=100)
    plt.close(fig)


# Compara la MLP contra los modelos de results/metrics.csv
def plot_comparison(table):
    colors = {'Random Forest': '#2a9d8f', 'XGBoost': '#e76f51', 'MLP': '#6a4c93'}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, metric in zip(axes, METRIC_NAMES):
        ax.barh(table.index, table[f'{metric}_test'],
                color=[colors.get(m, '#9aa5b1') for m in table.index])
        ax.set_title(f'{metric} (prueba)')
        ax.invert_yaxis()
        if metric != METRIC_NAMES[0]:
            ax.set_yticklabels([])
    axes[0].axvline(0, color='k', lw=0.8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'comparacion_metricas_mlp.png', dpi=100)
    plt.close(fig)


# La MLP no tiene feature_importances_, por eso se usa importancia por permutacion:
# cuanto sube el MAE en prueba al desordenar cada variable
def plot_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, scoring='neg_mean_absolute_error',
                                    n_repeats=10, random_state=SEED, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=FEATURES).sort_values()
    ax = importances.plot.barh(figsize=(7, 4.5), color='#6a4c93')
    ax.set_xlabel('Aumento del MAE al permutar la variable')
    ax.set_title('Importancia de variables - MLP')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'importancia_variables_mlp.png', dpi=100)
    plt.close()
    return importances.sort_values(ascending=False)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_clean_data()
    X_train, X_test, y_train, y_test, g_train, _ = group_train_test_split(
        X, y, groups, test_size=0.2, random_state=SEED)
    print(f'Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas')

    print('Ajustando hiperparametros (solo con entrenamiento)...')
    search = tune_mlp(X_train, y_train, g_train)
    params = {k.replace('regressor__model__', ''): v for k, v in search.best_params_.items()}
    print(f'MLP: {params}')

    grid = pd.DataFrame(search.cv_results_)
    grid = pd.DataFrame({
        'capas': grid['param_regressor__model__hidden_layer_sizes'].astype(str),
        'alpha': grid['param_regressor__model__alpha'],
        'learning_rate_init': grid['param_regressor__model__learning_rate_init'],
        'MAE_val': -grid['mean_test_score'],
    }).sort_values('MAE_val').round(4)
    grid.to_csv(RESULTS_DIR / 'hiperparametros_mlp.csv', index=False)

    print('Evaluando en entrenamiento/prueba y con validacion cruzada agrupada...')
    models = {'MLP': create_mlp(**params)}
    results = evaluate_models(models, X_train, y_train, X_test, y_test)
    _, cv_summary = cross_validate_grouped(models, X, y, groups, n_splits=5)

    test = results[results['Conjunto'] == 'test'].set_index('Modelo')[METRIC_NAMES]
    table = test.add_suffix('_test')
    table['R2_train'] = results[results['Conjunto'] == 'train'].set_index('Modelo')['R2']
    for metric in ['R2', 'RMSE', 'MAE']:
        table[f'{metric}_cv_mean'] = cv_summary[(metric, 'mean')]
        table[f'{metric}_cv_std'] = cv_summary[(metric, 'std')]
    table = table.round(4)
    table.reset_index().to_csv(RESULTS_DIR / 'metrics_mlp.csv', index=False)
    print(table.to_string())

    mlp = create_mlp(**params).fit(X_train, y_train)
    pred = mlp.predict(X_test)
    err = error_by_range(y_test, pred)

    others = pd.read_csv(ROOT / 'results' / 'metrics.csv').set_index('Modelo')
    comparison = pd.concat([others[table.columns], table])

    plot_learning_curve(mlp.regressor_.named_steps['model'])
    plot_predictions(y_test, pred)
    plot_comparison(comparison)
    importances = plot_permutation_importance(mlp, X_test, y_test)

    cols = [f'{m}_test' for m in METRIC_NAMES] + ['R2_train', 'R2_cv_mean', 'R2_cv_std']
    lines = [
        'MLP - Prediccion de ORGC (%) con AllpaDB',
        '=' * 50,
        f'Division 80/20 agrupada por perfil: {len(X_train)} filas de entrenamiento, {len(X_test)} de prueba.',
        f'Hiperparametros (CV agrupada de 3 folds, se minimiza el MAE): {params}',
        f'MAE de validacion: {-search.best_score_:.3f}',
        f'Epocas de entrenamiento: {mlp.regressor_.named_steps["model"].n_iter_}',
        '',
        'METRICAS EN EL CONJUNTO DE PRUEBA',
        comparison[cols].round(4).to_string(),
        '',
        'MAE POR RANGO DE ORGC REAL (prueba)',
        err.round(3).to_string(),
        f'Predicciones negativas: {(pred < 0).sum()}',
        '',
        'IMPORTANCIA POR PERMUTACION (aumento del MAE)',
        importances.round(3).to_string(),
    ]
    (RESULTS_DIR / 'mlp_comparison.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Resultados guardados en {RESULTS_DIR}')


if __name__ == '__main__':
    main()
