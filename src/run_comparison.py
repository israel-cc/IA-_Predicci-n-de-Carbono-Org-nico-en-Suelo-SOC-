"""Compara los modelos baseline contra Random Forest y XGBoost y genera los archivos de results/.

Uso (desde la raiz del repositorio):
    python src/run_comparison.py

Genera:
    results/metrics.csv            tabla comparativa de metricas
    results/plots/*.png            graficos de resultados
    results/model_comparison.txt   resumen numerico (el analisis detallado se complementa a mano)
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # sin ventana: solo guarda las figuras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import (METRIC_NAMES, cross_validate_grouped, evaluate_models,
                        filter_valid_target, group_train_test_split)
from models import (create_random_forest, create_xgboost, get_all_models,
                    tune_random_forest, tune_xgboost)
from preprocessing import FEATURES, load_data, prepare_data

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data' / 'raw' / 'allpaDB_layer_HARMONIZED_v1.csv'
RESULTS_DIR = ROOT / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'
SEED = 42

RF = 'Random Forest'
XGB = 'XGBoost'
ADVANCED = [RF, XGB]
BEST_LINEAR = 'Regresion Lineal'


def load_clean_data():
    df = load_data(DATA_PATH)
    X, y = prepare_data(df)
    groups = df.loc[X.index, 'profile_code']
    return filter_valid_target(X, y, groups)


# Tabla final: metricas de prueba, R2 de entrenamiento (para ver sobreajuste) y CV agrupada
def build_metrics_table(results, cv_summary, models):
    test = results[results['Conjunto'] == 'test'].set_index('Modelo')[METRIC_NAMES]
    train_r2 = results[results['Conjunto'] == 'train'].set_index('Modelo')['R2']
    table = test.add_suffix('_test').loc[list(models)]
    table['R2_train'] = train_r2
    for metric in ['R2', 'RMSE', 'MAE']:
        table[f'{metric}_cv_mean'] = cv_summary[(metric, 'mean')]
        table[f'{metric}_cv_std'] = cv_summary[(metric, 'std')]
    return table.round(4).reset_index()


def plot_metrics_bars(table):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    palette = {RF: '#2a9d8f', XGB: '#e76f51'}
    colors = [palette.get(m, '#9aa5b1') for m in table['Modelo']]
    for ax, metric in zip(axes, METRIC_NAMES):
        ax.barh(table['Modelo'], table[f'{metric}_test'], color=colors)
        ax.set_title(f'{metric} (prueba)')
        ax.invert_yaxis()
        if metric != METRIC_NAMES[0]:
            ax.set_yticklabels([])
    axes[0].axvline(0, color='k', lw=0.8)
    fig.suptitle('Comparacion de modelos en el conjunto de prueba '
                 '(R2: mas alto es mejor; RMSE, MAE, MedAE: mas bajo es mejor)')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'comparacion_metricas.png', dpi=150)
    plt.close(fig)


def plot_predictions(fitted, X_test, y_test):
    names = [BEST_LINEAR] + ADVANCED
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.5), sharey=True)
    lim = y_test.quantile(0.99)
    for ax, name in zip(axes, names):
        ax.scatter(y_test, fitted[name].predict(X_test), s=6, alpha=0.3)
        ax.plot([0, lim], [0, lim], 'r--', label='Prediccion perfecta')
        ax.set_xlim(0, lim)
        ax.set_ylim(-1, lim)
        ax.set_xlabel('ORGC real (%)')
        ax.set_title(name)
        ax.legend()
    axes[0].set_ylabel('ORGC predicho (%)')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'real_vs_predicho.png', dpi=150)
    plt.close(fig)


# MAE por rango de ORGC real: muestra donde falla cada modelo (valores altos vs bajos)
def error_by_range(fitted, X_test, y_test):
    bins = [0, 1, 3, 10, np.inf]
    labels = ['< 1 %', '1 - 3 %', '3 - 10 %', '> 10 %']
    ranges = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)
    rows = []
    for name in [BEST_LINEAR] + ADVANCED:
        abs_err = (y_test - fitted[name].predict(X_test)).abs()
        for label in labels:
            mask = (ranges == label).to_numpy()
            rows.append({'Modelo': name, 'Rango ORGC': label, 'n': int(mask.sum()),
                         'MAE': abs_err[mask].mean()})
    return pd.DataFrame(rows)


def plot_error_by_range(err):
    pivot = err.pivot(index='Rango ORGC', columns='Modelo', values='MAE')
    pivot = pivot.loc[err['Rango ORGC'].unique(), [BEST_LINEAR] + ADVANCED]
    ax = pivot.plot.bar(figsize=(8, 4.5), color=['#9aa5b1', '#2a9d8f', '#e76f51'], rot=0)
    ax.set_ylabel('MAE (puntos porcentuales de ORGC)')
    ax.set_title('Error absoluto medio segun el ORGC real (prueba)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'mae_por_rango.png', dpi=150)
    plt.close()


# Importancia de variables de los dos modelos de arboles (cada una normalizada a suma 1)
def plot_feature_importance(fitted):
    importances = pd.DataFrame({
        name: fitted[name].named_steps['model'].feature_importances_ for name in ADVANCED
    }, index=FEATURES)
    importances = importances / importances.sum()
    importances = importances.sort_values(RF)
    ax = importances.plot.barh(figsize=(8, 5), color=['#2a9d8f', '#e76f51'])
    ax.set_xlabel('Importancia relativa')
    ax.set_title('Importancia de variables - Random Forest vs XGBoost')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'importancia_variables.png', dpi=150)
    plt.close()
    return importances.sort_values(RF, ascending=False)


# Analisis detallado. Las cifras corresponden a la ejecucion con SEED = 42 y la rejilla de
# models.RF_PARAM_GRID y models.XGB_PARAM_GRID; si se cambia la semilla, los datos o el modelo, hay que revisarlo.
ANALYSIS = """ANALISIS DETALLADO
------------------
1. Que modelo funciono mejor
   Los dos modelos de arboles superan claramente a la regresion lineal (R2 = 0.09), pero entre si
   estan practicamente empatados:
   - Division unica de prueba: Random Forest R2 = 0.28 y RMSE = 4.33; XGBoost R2 = 0.23 y
     RMSE = 4.47.
   - Validacion cruzada agrupada por perfil (5 folds): R2 = 0.285 +- 0.041 (Random Forest) y
     0.282 +- 0.037 (XGBoost). La diferencia es mucho menor que la desviacion entre folds, asi
     que la ventaja de Random Forest en la division unica se debe en buena parte a que perfiles
     cayeron en prueba, y no a que sea mejor modelo.
   Conclusion: Random Forest es el mejor por una pequena diferencia y es la opcion mas estable;
   XGBoost es equivalente en promedio. Ninguno de los dos domina en todo el rango de ORGC:
   XGBoost es mejor en suelos > 10 % (MAE 16.4 frente a 17.6) y en MedAE (0.53 frente a 0.56);
   Random Forest es mejor en 1-3 % y 3-10 % (MAE 1.38 y 2.80 frente a 1.50 y 3.13).

   Tampoco ganan en todas las metricas. En MAE y MedAE la mediana constante (1.64 y 0.49) y la
   regresion lineal sobre log(ORGC) (1.54 y 0.52) igualan o superan a ambos (Random Forest 1.63 y
   0.56; XGBoost 1.66 y 0.53). La razon es que ORGC esta muy concentrada en valores bajos (mediana
   de 0.69 %, media de 2.0 %): predecir siempre un valor bajo acierta en la mayoria de los suelos.
   Los modelos de arboles minimizan el error cuadratico, asi que suben sus predicciones para
   cubrir los suelos ricos en carbono, y eso les cuesta algo de error en los suelos tipicos. Es
   un compromiso entre acertar lo comun (MAE/MedAE) y reducir las fallas grandes (RMSE/R2).

2. Por que los modelos de arboles mejoran a los lineales
   Capturan relaciones no lineales e interacciones que la regresion lineal no puede representar:
   el carbono depende de la profundidad de la capa, del pH y de la capacidad de intercambio
   cationico (ECEC) de forma no proporcional. Los dos modelos coinciden en las variables mas
   importantes (PHAQ, lower_depth y ECEC suman ~50 % con Random Forest y ~42 % con XGBoost),
   coherente con la literatura sobre propiedades edaficas. Tampoco necesitan escalar variables.

3. Por que XGBoost no supera a Random Forest aqui
   El paper de la costa norte de Peru reporta XGBoost como el mejor modelo, pero con predictores
   ambientales (vegetacion, clima, relieve) y solo 109 muestras. Aqui hay 21,601 filas pero solo
   9 predictores edaficos; el limite lo pone la informacion disponible, no el algoritmo.
   XGBoost, al corregir errores de forma secuencial, se ajusta mas al ruido: R2 de 0.83 en
   entrenamiento frente a 0.23 en prueba (Random Forest: 0.53 frente a 0.28). Random Forest, al
   promediar arboles independientes, es mas robusto cuando los datos son ruidosos.

4. Limitaciones
   - Sigue explicando poca varianza (R2 ~ 0.28): solo hay 9 predictores (profundidad, pH,
     textura, carbonatos, conductividad y ECEC). Los papers revisados que logran R2 altos usan
     ademas clima, vegetacion (NDVI, EVI), relieve y ubicacion, que aqui no estan.
   - Sobreajuste: fuerte en XGBoost (0.83 frente a 0.23) y moderado en Random Forest.
   - Suelos con ORGC > 10 % (turbosos u organicos): son pocos (137 de 4,252 en prueba), pero
     todos los modelos fallan mucho (MAE de 16-22). Dominan el RMSE y el R2.
   - En el rango 3-10 % ambos modelos de arboles tienen mas error que la regresion lineal.
   - La division agrupa por perfil, pero perfiles cercanos geograficamente pueden parecerse
     (autocorrelacion espacial), asi que el desempeno en una zona nueva podria ser menor.
   - La imputacion por mediana (ELCOSP tiene 36 % de faltantes y ECEC 23 %) puede restar
     informacion.
   - Los hiperparametros se ajustaron con una rejilla pequena; una busqueda mas amplia podria
     cambiar la diferencia entre ambos modelos.
   - ORGC proviene de Walkley-Black, mientras que varios papers usan combustion seca: los
     valores no son estrictamente comparables.

5. Siguientes pasos sugeridos
   - Evaluar una red neuronal MLP, tambien recomendada por la literatura.
   - Probar los arboles con criterio de error absoluto o sobre log(ORGC), para mejorar MAE.
   - Incorporar covariables ambientales (clima, vegetacion, relieve) a partir de las coordenadas.
   - Tratar aparte los suelos organicos, o reportar el error por rango de ORGC."""


def write_summary(table, tuned, err, importances, n_train, n_test):
    t = table.set_index('Modelo')
    best_rmse = t['RMSE_test'].idxmin()
    best_mae = t['MAE_test'].idxmin()
    lines = [
        'COMPARACION DE MODELOS - Prediccion de ORGC (%) con AllpaDB',
        '=' * 62,
        f'Division 80/20 agrupada por perfil: {n_train} filas de entrenamiento, {n_test} de prueba.',
        'Hiperparametros (CV agrupada de 3 folds sobre entrenamiento, se minimiza el MAE):',
        *[f'  {name}: {params} | MAE de validacion {mae:.3f}' for name, (params, mae) in tuned.items()],
        '',
        'METRICAS EN EL CONJUNTO DE PRUEBA',
        t[[f'{m}_test' for m in METRIC_NAMES] + ['R2_train', 'R2_cv_mean', 'R2_cv_std']].to_string(),
        '',
        'MAE POR RANGO DE ORGC REAL (prueba)',
        'Muestras por rango: ' + ', '.join(
            f'{r.Index} = {r.n}' for r in err[err['Modelo'] == RF].set_index('Rango ORGC').itertuples()),
        err.pivot(index='Rango ORGC', columns='Modelo', values='MAE').loc[err['Rango ORGC'].unique()].round(3).to_string(),
        '',
        'IMPORTANCIA DE VARIABLES (relativa, suma 1)',
        importances.round(3).to_string(),
        '',
        'RESUMEN',
        f'- Mejor R2: {t["R2_test"].idxmax()} ({t["R2_test"].max():.3f}). Mejor RMSE: {best_rmse} ({t.loc[best_rmse, "RMSE_test"]:.2f}).',
        f'- Mejor MAE: {best_mae} ({t.loc[best_mae, "MAE_test"]:.2f}). Mejor MedAE: {t["MedAE_test"].idxmin()} ({t["MedAE_test"].min():.2f}).',
        *[f'- {m}: R2 train {t.loc[m, "R2_train"]:.3f} vs test {t.loc[m, "R2_test"]:.3f}; '
          f'CV R2 {t.loc[m, "R2_cv_mean"]:.3f} +- {t.loc[m, "R2_cv_std"]:.3f}.' for m in ADVANCED],
        '',
        'POR QUE ESTAS METRICAS',
        'ORGC es una concentracion en %, muy asimetrica: casi todos los suelos tienen valores bajos y',
        'pocos (organicos o turbosos) superan 10 %. Por eso ninguna metrica sola es suficiente:',
        '- MAE: error promedio en puntos porcentuales de ORGC; es la metrica principal por ser directa',
        '  de interpretar y poco dominada por los valores extremos.',
        '- MedAE: error de la prediccion tipica; refleja el desempeno en la gran mayoria de suelos.',
        '- RMSE: eleva el error al cuadrado, asi que penaliza las fallas grandes en suelos ricos en',
        '  carbono, que son los de mayor interes para stocks de carbono. MAE y RMSE juntos muestran si',
        '  un modelo acierta en lo comun pero falla en lo extremo.',
        '- R2: fraccion de la varianza explicada; permite comparar con los papers revisados.',
        'Todas se calculan en la escala original de ORGC (%), no en escala log, para ser comparables',
        'entre modelos. La exactitud (accuracy) o el F1 no aplican: es un problema de regresion.',
    ]
    (RESULTS_DIR / 'model_comparison.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_clean_data()
    X_train, X_test, y_train, y_test, g_train, _ = group_train_test_split(
        X, y, groups, test_size=0.2, random_state=SEED)
    print(f'Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas')

    print('Ajustando hiperparametros (solo con entrenamiento)...')
    tuned, estimators = {}, {}
    for name, tune, create in [(RF, tune_random_forest, create_random_forest),
                               (XGB, tune_xgboost, create_xgboost)]:
        search = tune(X_train, y_train, g_train, random_state=SEED)
        params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
        tuned[name] = (params, -search.best_score_)
        estimators[name] = create(random_state=SEED, **params)
        print(f'{name}: {params}')

    models = get_all_models(random_forest=estimators[RF], xgboost=estimators[XGB])

    print('Evaluando en entrenamiento/prueba y con validacion cruzada agrupada...')
    results = evaluate_models(models, X_train, y_train, X_test, y_test)
    _, cv_summary = cross_validate_grouped(models, X, y, groups, n_splits=5)
    table = build_metrics_table(results, cv_summary, models)
    table.to_csv(RESULTS_DIR / 'metrics.csv', index=False)
    print(table.to_string(index=False))

    fitted = {name: model.fit(X_train, y_train) for name, model in models.items()}
    plot_metrics_bars(table)
    plot_predictions(fitted, X_test, y_test)
    err = error_by_range(fitted, X_test, y_test)
    plot_error_by_range(err)
    importances = plot_feature_importance(fitted)

    write_summary(table, tuned, err, importances, len(X_train), len(X_test))
    print(f'Resultados guardados en {RESULTS_DIR}')


if __name__ == '__main__':
    main()
