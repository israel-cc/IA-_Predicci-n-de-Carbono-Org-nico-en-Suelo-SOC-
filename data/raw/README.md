# Dataset: AllpaDB (data/raw)

## 1. Origen

**AllpaDB** es una base de datos que compila y armoniza información de estudios de suelos del Perú realizados desde 1962 (datos legados y de investigación), que antes no estaba digitalizada ni estructurada.

- **Título del dataset:** AllpaDB Dataset: Harmonized soil legacy and research data from Peru (Versión 1)
- **Autores:** Carlos Mestanza (contacto), Renzo Huerta, Riguel Llerena, Oscar Panduro, Miguel Culqui, Cristhian Oscco, Melissa Quispe, Lily Tello, Julio Nazario (Universidad Nacional Agraria La Molina), Alexandre ten Caten (Universidade Federal de Santa Catarina) y Cesar Beltrán (Pontificia Universidad Católica del Perú).
- **Repositorio:** Zenodo, publicado el 8 de mayo de 2026. DOI: [10.5281/zenodo.20077332](https://doi.org/10.5281/zenodo.20077332)
- **Artículo asociado:** _AllpaDB: a compiled repository of harmonized soil legacy data and research data from Peru_, Scientific Data. DOI: [10.1038/s41597-026-07558-9](https://doi.org/10.1038/s41597-026-07558-9)

## 2. Permisos y licencia

- **Licencia:** Creative Commons Attribution 4.0 International (**CC BY 4.0**). Permite redistribuir y reutilizar los datos, incluso con fines académicos, siempre que se cite a los autores.
- **Acceso:** abierto.
- **Cita sugerida** : Mestanza, C., Huerta, R., Llerena, R. et al. AllpaDB: a compiled repository of harmonized soil legacy data and research data from Peru. Sci Data (2026). https://doi.org/10.1038/s41597-026-07558-9

## 3. Archivos en este repositorio

| Archivo                           | Descripción                                                                                               |
| --------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `allpaDB_layer_HARMONIZED_v1.csv` | Datos por capa de suelo con las variables de laboratorio armonizadas. Es el archivo usado en el proyecto. |
| `allpaDB_dictionary_v1.csv`       | Diccionario de variables: código, descripción, tipo de dato y unidades.                                   |

Zenodo contiene otros archivos que no se incluyen aquí (versiones RAW y CORRECTED de las capas, tabla de perfiles, descripción de métodos de laboratorio y capas GIS de gran tamaño).

## 4. Características del archivo `allpaDB_layer_HARMONIZED_v1.csv`

- **Unidad de observación:** una capa (horizonte) de un perfil de suelo.
- **Tamaño:** 24,341 capas y 45 columnas.
- **Perfiles:** 8,793 identificadores de perfil distintos (`profile_code`), provenientes de 224 documentos fuente (`doc_code`). Varias capas pertenecen al mismo perfil, por lo que no son observaciones independientes.
- **Variable objetivo:** `ORGC`, carbono orgánico total por oxidación húmeda de Walkley & Black (%). Tiene 21,609 valores presentes (11.2 % faltantes), media de 2.06, mediana de 0.69 y distribución muy asimétrica a la derecha. Hay 8 valores mayores a 100 %, que son físicamente imposibles.
- **Variables predictoras usadas en el proyecto:** `upper_depth`, `lower_depth`, `PHAQ`, `SAND`, `SILT`, `CLAY`, `TCEQ`, `ELCOSP`, `ECEC`. Los valores faltantes van desde 4 % (profundidades) hasta 38 % (`ELCOSP`).
- **Otros grupos de variables:** capacidad de intercambio catiónico, conductividad eléctrica, pH en distintos medios, fósforo, nitrógeno, densidad aparente, fracciones de arena, bases intercambiables, color Munsell, estructura, además de pendiente, precipitación, temperatura, zona de vida, material parental y clasificación (ST y WRB).
- **Ubicación:** el diccionario define `latitude` y `longitude`, pero esas columnas no están en este archivo de capas (de ahí las 45 columnas, frente a las 64 variables del diccionario).

Las definiciones completas de cada variable están en `allpaDB_dictionary_v1.csv`.

## 5. Limitaciones

- Datos legados provenientes de distintos laboratorios y épocas, armonizados pero con posibles inconsistencias.
- Alta proporción de valores faltantes en varias variables (desbajance de cantidad de datos).
- La representatividad geográfica no se ha verificado en este proyecto, porque el archivo de capas no incluye coordenadas.
