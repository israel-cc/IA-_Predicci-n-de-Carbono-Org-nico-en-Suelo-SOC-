import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURES = ['upper_depth', 'lower_depth', 'PHAQ', 'SAND', 'SILT', 'CLAY', 'TCEQ', 'ELCOSP', 'ECEC']
TARGET = 'ORGC'

# Carga de datos desde un archivo CSV
def load_data(path):
    return pd.read_csv(path)

# Separando las variables predictoras y la variable objetivo
def prepare_data(df):

    data = df[FEATURES + [TARGET]].copy()

    # El target no puede quedar ausente para entrenar un modelo
    data = data.dropna(subset=[TARGET])

    X = data[FEATURES]
    y = data[TARGET]

    return X, y

# Creando el pipeline de preprocesamiento: imputación y escalamiento
def create_preprocessor_scaled():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

# Creando el pipeline de preprocesamiento: solo imputación, sin escalamiento
def create_preprocessor_unscaled():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])