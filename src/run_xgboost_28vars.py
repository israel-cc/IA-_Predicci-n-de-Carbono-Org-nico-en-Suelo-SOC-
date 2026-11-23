"""XGBoost con las 28 variables del Random Forest ampliado (notebooks/Random_forest_Huaroto.ipynb).

Uso (desde la raiz del repositorio):
    python src/run_xgboost_28vars.py

Se agregan a las 8 variables de data/processed/ las mismas 20 variables nuevas del notebook de
Random Forest (horizonte, cationes intercambiables, acidez, CIC a pH 7, color Munsell, posicion de
la capa y fuente), con la misma division por perfil (random_state=42) y la misma validacion
interna. BD2 y NITKJD se excluyen por fuga de informacion, igual que en ese notebook.

Se entrena sobre ORGC_log = log1p(ORGC). Los hiperparametros se eligen en validacion (con early
stopping); el modelo final se reentrena con train + validacion y se evalua una sola vez en test.
Como referencia se reentrena el Random Forest de 28 variables con sus hiperparametros finales.

Genera results/xgboost_28vars/metrics_xgboost_28vars.csv, xgboost_28vars.txt y plots/.
"""
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / 'data' / 'processed'
RAW = ROOT / 'data' / 'raw' / 'allpaDB_layer_HARMONIZED_v1.csv'
OUT = ROOT / 'results' / 'xgboost_28vars'
SEED = 42

NUEVAS_NUM = ['ECA', 'EMG', 'EK', 'ENA', 'EAH', 'CECPH7', 'm_value', 'm_chroma', 'rank']
HZ_COLS = ['hz_A', 'hz_B', 'hz_C', 'hz_O', 'hz_H', 'hz_E', 'hz_R', 'hz_X']
SRC_COLS = ['src_HD', 'src_RES', 'src_ZEE']


# ---------------------------------------------------------------- datos (igual que el notebook de RF)
def build_features():
    X_train_base = pd.read_csv(PROC / 'X_train.csv')
    X_test_base = pd.read_csv(PROC / 'X_test.csv')
    y_train_df = pd.read_csv(PROC / 'y_train.csv')
    y_test_df = pd.read_csv(PROC / 'y_test.csv')

    df = pd.read_csv(RAW)
    df = df[df['ORGC'].notnull() & (df['ORGC'] <= 100)].copy()

    # Horizonte: primera letra de layer_name sin numeros/parentesis iniciales ("2Bt" -> "B")
    hz = df['layer_name'].fillna('').str.replace(r'^[\d\(\)\s]+', '', regex=True).str[:1].str.upper()
    df['horizonte'] = hz.where(hz.isin(['A', 'B', 'C', 'O', 'H', 'E', 'R']), 'X')
    # Color Munsell humedo: "10YR 4/3" -> value 4, chroma 3
    munsell = df['MSTCOL'].fillna('').str.extract(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)').astype(float)
    df['m_value'], df['m_chroma'] = munsell[0], munsell[1]
    # Posicion de la capa dentro del perfil
    df['rank'] = df.groupby('profile_code')['upper_depth'].rank(method='first')

    # Misma division por perfil que data/processed/
    profiles = df['profile_code'].unique()
    train_profiles, _ = train_test_split(profiles, test_size=0.2, random_state=SEED)
    is_train = df['profile_code'].isin(train_profiles)
    train_raw = df[is_train].reset_index(drop=True)
    test_raw = df[~is_train].reset_index(drop=True)
    assert np.allclose(train_raw['ORGC'], y_train_df['ORGC']), 'train no coincide con data/processed'
    assert np.allclose(test_raw['ORGC'], y_test_df['ORGC']), 'test no coincide con data/processed'

    # Imputador ajustado solo con train
    imputer = SimpleImputer(strategy='median').fit(train_raw[NUEVAS_NUM])

    def agregar(raw, base):
        num = pd.DataFrame(imputer.transform(raw[NUEVAS_NUM]), columns=NUEVAS_NUM)
        h = pd.get_dummies(raw['horizonte'], prefix='hz').astype(int).reindex(columns=HZ_COLS, fill_value=0)
        s = pd.get_dummies(raw['font_type'], prefix='src').astype(int).reindex(columns=SRC_COLS, fill_value=0)
        return pd.concat([base, h, s, num], axis=1)

    X_train = agregar(train_raw, X_train_base)
    X_test = agregar(test_raw, X_test_base)
    return (X_train, X_test, y_train_df, y_test_df,
            train_raw['profile_code'].values, train_profiles, list(X_train_base.columns))


# ---------------------------------------------------------------- utilidades
def metrics(y_true, y_pred):
    return {'R2': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MedAE': median_absolute_error(y_true, y_pred)}


def xgb(**params):
    return XGBRegressor(objective='reg:squarederror', tree_method='hist', subsample=0.8,
                        random_state=SEED, n_jobs=-1, **params)


def main():
    (OUT / 'plots').mkdir(parents=True, exist_ok=True)
    X_full, X_test, y_train_df, y_test_df, g_train, train_profiles, cols_base = build_features()
    print(f'X_train {X_full.shape} | X_test {X_test.shape}')

    y_full_log = y_train_df['ORGC_log']
    y_test_log, y_test_pct = y_test_df['ORGC_log'], y_test_df['ORGC']

    # Validacion interna: 20 % de los perfiles de train (igual que el notebook de RF)
    tr_p, val_p = train_test_split(train_profiles, test_size=0.2, random_state=SEED)
    m_tr, m_val = np.isin(g_train, tr_p), np.isin(g_train, val_p)
    X_tr, X_val, y_tr, y_val = X_full[m_tr], X_full[m_val], y_full_log[m_tr], y_full_log[m_val]

    # Busqueda de hiperparametros en validacion, con early stopping para el numero de arboles
    grid = {'max_depth': [4, 6, 8, 10], 'min_child_weight': [1, 5],
            'colsample_bytree': [0.5, 0.8], 'reg_lambda': [1, 10]}
    rows = []
    for md, mcw, cs, rl in product(*grid.values()):
        m = xgb(n_estimators=3000, learning_rate=0.03, max_depth=md, min_child_weight=mcw,
                colsample_bytree=cs, reg_lambda=rl, early_stopping_rounds=100, eval_metric='rmse')
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        pv = m.predict(X_val)
        rows.append({'max_depth': md, 'min_child_weight': mcw, 'colsample_bytree': cs,
                     'reg_lambda': rl, 'n_estimators': m.best_iteration + 1,
                     'rmse_val': np.sqrt(mean_squared_error(y_val, pv)), 'r2_val': r2_score(y_val, pv)})
        print(rows[-1])
    search = pd.DataFrame(rows).sort_values('rmse_val').reset_index(drop=True)
    best = search.iloc[0]
    best_params = {'max_depth': int(best['max_depth']), 'min_child_weight': int(best['min_child_weight']),
                   'colsample_bytree': float(best['colsample_bytree']), 'reg_lambda': float(best['reg_lambda']),
                   # se escala el numero de arboles porque el modelo final ve un 25 % mas de datos
                   'n_estimators': int(round(best['n_estimators'] * 1.25)), 'learning_rate': 0.03}
    print('Mejores hiperparametros:', best_params)

    # Modelos finales: entrenados con train + validacion, evaluados una vez en test
    models = {
        'XGBoost 8 variables (mismos hiperparametros)': (xgb(**best_params), cols_base),
        'Random Forest 28 variables (Huaroto)': (RandomForestRegressor(
            n_estimators=500, max_features=0.5, max_depth=25, min_samples_leaf=1,
            random_state=SEED, n_jobs=-1), list(X_full.columns)),
        'XGBoost 28 variables': (xgb(**best_params), list(X_full.columns)),
    }
    table, preds = [], {}
    for name, (model, cols) in models.items():
        model.fit(X_full[cols], y_full_log)
        p_log = model.predict(X_test[cols])
        p_pct = np.expm1(p_log)
        preds[name] = p_pct
        r = {'Modelo': name, 'n_variables': len(cols)}
        r.update({f'{k}_log': v for k, v in metrics(y_test_log, p_log).items()})
        r.update({f'{k}_pct': v for k, v in metrics(y_test_pct, p_pct).items()})
        r['R2_train_log'] = r2_score(y_full_log, model.predict(X_full[cols]))
        err = np.abs(y_test_pct - p_pct)
        for t in [0.25, 0.5, 1]:
            r[f'acierto_{t}'] = (err <= t).mean() * 100
        table.append(r)
        if name == 'XGBoost 28 variables':
            final_model = model
    table = pd.DataFrame(table).round(4)
    table.to_csv(OUT / 'metrics_xgboost_28vars.csv', index=False)
    search.round(4).to_csv(OUT / 'busqueda_hiperparametros.csv', index=False)

    # Error por rango e importancia de variables (ganancia)
    rangos = pd.cut(y_test_pct, [0, 1, 3, 10, 100], labels=['< 1 %', '1 - 3 %', '3 - 10 %', '> 10 %'],
                    include_lowest=True)
    err_rango = pd.DataFrame({'n': y_test_pct.groupby(rangos, observed=False).size()})
    for name, p in preds.items():
        err_rango[f'MAE {name}'] = np.abs(y_test_pct - p).groupby(rangos, observed=False).mean().round(3)
    imp = pd.Series(final_model.feature_importances_, index=X_full.columns).sort_values(ascending=False)

    with open(OUT / 'xgboost_28vars.txt', 'w', encoding='utf-8') as f:
        f.write('XGBoost con 28 variables (mismas variables que el Random Forest ampliado)\n' + '=' * 70 + '\n')
        f.write(f'Train {len(X_full)} filas (validacion interna {m_val.sum()}), test {len(X_test)} filas.\n')
        f.write(f'Hiperparametros (validacion por perfil, early stopping): {best_params}\n\n')
        f.write('TOP 5 DE LA BUSQUEDA (validacion, escala log)\n' + search.head().round(4).to_string() + '\n\n')
        f.write('METRICAS EN TEST (_log: escala log1p; _pct: % de ORGC; acierto_t: % de capas con error <= t)\n')
        f.write(table.set_index('Modelo').T.to_string() + '\n\n')
        f.write('MAE POR RANGO DE ORGC REAL (test, %)\n' + err_rango.T.to_string() + '\n\n')
        f.write('IMPORTANCIA DE VARIABLES (ganancia, XGBoost 28 variables)\n' + imp.round(4).to_string() + '\n')
    print(open(OUT / 'xgboost_28vars.txt', encoding='utf-8').read())

    # Graficos
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, name in zip(axes[:2], ['XGBoost 8 variables (mismos hiperparametros)', 'XGBoost 28 variables']):
        p_log = np.log1p(preds[name])
        ax.scatter(y_test_log, p_log, s=6, alpha=0.3, color='#e76f51')
        lim = [y_test_log.min(), y_test_log.max()]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set(xlabel='ORGC_log real', ylabel='ORGC_log predicho',
               title=f'{name}\nR2 (log) = {r2_score(y_test_log, p_log):.3f}')
    imp.head(15).sort_values().plot.barh(ax=axes[2], color='#e76f51')
    axes[2].set(title='Importancia (ganancia) - top 15')
    plt.tight_layout()
    plt.savefig(OUT / 'plots' / 'xgboost_28vars.png', dpi=130)


if __name__ == '__main__':
    main()
