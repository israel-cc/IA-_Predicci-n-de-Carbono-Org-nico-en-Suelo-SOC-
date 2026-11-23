"""Comparacion unificada de todos los modelos con las 28 variables.

Uso (desde la raiz del repositorio):
    python src/run_all_28vars.py

Todos los modelos usan las mismas 28 variables (src/run_xgboost_28vars.py), la misma division
por perfil y el mismo objetivo ORGC_log = log1p(ORGC). Las predicciones se devuelven a % con
expm1. Hiperparametros: los elegidos por cada integrante (RF: notebook de Huaroto; XGBoost:
run_xgboost_28vars.py; Cubist y MLP: reajustados con 28 variables en src/run_cv_28vars.py). Los modelos estan en src/models.py.

Genera results/final_28vars/metrics_28vars.csv, comparacion_28vars.txt y plots/.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models import create_cubist_28, create_mlp_28, create_random_forest_28, create_xgboost_28
from run_xgboost_28vars import SEED, build_features, metrics, xgb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'final_28vars'

XGB_PARAMS = {'max_depth': 10, 'min_child_weight': 1, 'colsample_bytree': 0.8, 'reg_lambda': 10.0,
              'n_estimators': 1009, 'learning_rate': 0.03}


def get_models():
    return {
        'Media': DummyRegressor(strategy='mean'),
        'Mediana': DummyRegressor(strategy='median'),
        'Regresion Lineal': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        'MLP': create_mlp_28(random_state=SEED),
        'Cubist': create_cubist_28(random_state=SEED),
        'Random Forest': create_random_forest_28(random_state=SEED),
        'XGBoost': create_xgboost_28(random_state=SEED),
    }


def main():
    (OUT / 'plots').mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train_df, y_test_df, _, _, _ = build_features()
    y_train_log, y_test_log, y_test_pct = y_train_df['ORGC_log'], y_test_df['ORGC_log'], y_test_df['ORGC']
    print(f'X_train {X_train.shape} | X_test {X_test.shape}')

    rows, preds = [], {}
    for name, model in get_models().items():
        print('Entrenando', name)
        model.fit(X_train, y_train_log)
        p_log = model.predict(X_test)
        p_pct = np.clip(np.expm1(p_log), 0, 100)
        preds[name] = p_pct
        r = {'Modelo': name}
        r.update({f'{k}_log': v for k, v in metrics(y_test_log, p_log).items()})
        r.update({f'{k}_pct': v for k, v in metrics(y_test_pct, p_pct).items()})
        r['R2_train_log'] = r2_score(y_train_log, model.predict(X_train))
        err = np.abs(y_test_pct - p_pct)
        for t in [0.25, 0.5, 1]:
            r[f'acierto_{t}'] = (err <= t).mean() * 100
        rows.append(r)
    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT / 'metrics_28vars.csv', index=False)

    rangos = pd.cut(y_test_pct, [0, 1, 3, 10, 100], labels=['< 1 %', '1 - 3 %', '3 - 10 %', '> 10 %'],
                    include_lowest=True)
    err_rango = pd.DataFrame({name: np.abs(y_test_pct - p).groupby(rangos, observed=False).mean()
                              for name, p in preds.items()}).T.round(3)
    err_rango.loc['n capas'] = y_test_pct.groupby(rangos, observed=False).size()

    with open(OUT / 'comparacion_28vars.txt', 'w', encoding='utf-8') as f:
        f.write('Comparacion unificada con 28 variables (objetivo log1p(ORGC))\n' + '=' * 62 + '\n')
        f.write(f'Train {len(X_train)} filas, test {len(X_test)} filas (division por perfil).\n\n')
        f.write('METRICAS EN TEST (_log: escala log1p; _pct: % de ORGC; acierto_t: % de capas con error <= t)\n')
        f.write(table.set_index('Modelo').to_string() + '\n\n')
        f.write('MAE POR RANGO DE ORGC REAL (test, %)\n' + err_rango.to_string() + '\n')
    print(open(OUT / 'comparacion_28vars.txt', encoding='utf-8').read())
    err_rango.to_csv(OUT / 'mae_por_rango_28vars.csv')

    # Grafico 0: MAE por rango de ORGC (escala log en el eje y)
    er = err_rango.drop(['n capas', 'Mediana'], errors='ignore')
    n = err_rango.loc['n capas'].astype(int)
    fig, ax = plt.subplots(figsize=(9, 4))
    palette = ['#adb5bd', '#8d99ae', '#7b2cbf', '#e9c46a', '#2a9d8f', '#e76f51']
    w = 0.8 / len(er)
    for i, (name, row) in enumerate(er.iterrows()):
        ax.bar(np.arange(len(row)) + i * w - 0.4 + w / 2, row.values, w, label=name,
               color=palette[i % len(palette)])
    ax.set_yscale('log')
    ax.set_xticks(range(len(er.columns)))
    ax.set_xticklabels([f'{c}\n(n={n[c]})' for c in er.columns])
    ax.set(ylabel='MAE (% de ORGC, escala log)')
    ax.legend(ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / 'plots' / 'mae_por_rango_28vars.png', dpi=150)

    # Grafico 1: R2 y MAE por modelo
    t = table.set_index('Modelo')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ['#2a9d8f' if m == t['MAE_pct'].idxmin() else '#adb5bd' for m in t.index]
    t['R2_log'].plot.barh(ax=axes[0], color=colors)
    axes[0].set(title='R2 en test (escala log)', xlabel='R2')
    t['MAE_pct'].plot.barh(ax=axes[1], color=colors)
    axes[1].set(title='MAE en test (% de ORGC)', xlabel='MAE')
    for ax in axes:
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT / 'plots' / 'comparacion_metricas_28vars.png', dpi=150)

    # Grafico 2: real vs predicho de los tres mejores
    top = t.sort_values('R2_log', ascending=False).index[:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, top):
        p = np.log1p(preds[name])
        ax.scatter(y_test_log, p, s=5, alpha=0.3, color='#2a9d8f')
        lim = [y_test_log.min(), y_test_log.max()]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set(title=f'{name} (R2 log = {r2_score(y_test_log, p):.3f})',
               xlabel='log1p(ORGC) real', ylabel='log1p(ORGC) predicho')
    plt.tight_layout()
    plt.savefig(OUT / 'plots' / 'real_vs_predicho_28vars.png', dpi=150)


if __name__ == '__main__':
    main()
