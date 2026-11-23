"""Reajuste de Cubist y MLP con 28 variables y validacion cruzada agrupada de todos los modelos.

Uso (desde la raiz del repositorio):
    python src/run_cv_28vars.py

1) Cubist y MLP se reajustan con las 28 variables en la misma validacion interna que RF y XGBoost
   (20 % de los perfiles de entrenamiento).
2) Todos los modelos se evaluan con GroupKFold (5 folds por perfil) sobre el entrenamiento, para
   comprobar que el orden de la prueba no depende de una sola particion. El imputador de las
   variables nuevas se ajusta con todo el entrenamiento (igual que en la prueba).

Genera results/final_28vars/busqueda_cubist_mlp.csv y cv_28vars.csv.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from cubist import Cubist
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_all_28vars import XGB_PARAMS
from run_xgboost_28vars import SEED, build_features, xgb

OUT = Path(__file__).resolve().parent.parent / 'results' / 'final_28vars'


def mlp(layers, alpha):
    return Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(
        hidden_layer_sizes=layers, alpha=alpha, learning_rate_init=0.001, batch_size=256,
        early_stopping=True, n_iter_no_change=20, max_iter=500, random_state=SEED))])


def main():
    X, _, y_df, _, groups, train_profiles, _ = build_features()
    y = y_df['ORGC_log']

    # 1) Busqueda en validacion para Cubist y MLP
    tr_p, val_p = train_test_split(train_profiles, test_size=0.2, random_state=SEED)
    m_tr, m_val = np.isin(groups, tr_p), np.isin(groups, val_p)
    cands = [('Cubist', f'n_committees={c}, neighbors={n}',
              lambda c=c, n=n: Cubist(n_rules=100, n_committees=c, neighbors=n, random_state=SEED))
             for c, n in product([10, 20], [None, 5, 9])]
    cands += [('MLP', f'capas={l}, alpha={a}', lambda l=l, a=a: mlp(l, a))
              for l, a in product([(128, 64, 32), (256, 128, 64)], [0.01, 0.1, 1.0])]
    rows = []
    for name, desc, make in cands:
        m = make().fit(X[m_tr], y[m_tr])
        p = m.predict(X[m_val])
        rows.append({'Modelo': name, 'config': desc, 'r2_val': r2_score(y[m_val], p),
                     'rmse_val': np.sqrt(mean_squared_error(y[m_val], p)), 'make': make})
        print(rows[-1]['Modelo'], desc, round(rows[-1]['r2_val'], 4), flush=True)
    search = pd.DataFrame(rows).sort_values('rmse_val')
    search.drop(columns='make').round(4).to_csv(OUT / 'busqueda_cubist_mlp.csv', index=False)
    best = search.groupby('Modelo').head(1).set_index('Modelo')
    print(best[['config', 'r2_val']], flush=True)

    # 2) Validacion cruzada agrupada por perfil (5 folds)
    models = {
        'Regresion Lineal': lambda: Pipeline([('s', StandardScaler()), ('m', LinearRegression())]),
        'MLP': best.loc['MLP', 'make'],
        'Cubist': best.loc['Cubist', 'make'],
        'Random Forest': lambda: RandomForestRegressor(n_estimators=500, max_features=0.5, max_depth=25,
                                                       random_state=SEED, n_jobs=-1),
        'XGBoost': lambda: xgb(**XGB_PARAMS),
    }
    y_pct = y_df['ORGC']
    res = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        for name, make in models.items():
            p = make().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            pp = np.clip(np.expm1(p), 0, 100)
            res.append({'Modelo': name, 'fold': k, 'R2_log': r2_score(y.iloc[te], p),
                        'R2_pct': r2_score(y_pct.iloc[te], pp), 'MAE_pct': mean_absolute_error(y_pct.iloc[te], pp)})
            print(res[-1], flush=True)
    cv = pd.DataFrame(res).groupby('Modelo')[['R2_log', 'R2_pct', 'MAE_pct']].agg(['mean', 'std']).round(4)
    cv = cv.loc[list(models)]
    cv.to_csv(OUT / 'cv_28vars.csv')
    print(cv.to_string())
    print('Mejores configuraciones:\n', best[['config', 'r2_val']].to_string())


if __name__ == '__main__':
    main()
