"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------
En este laboratio se construirá un modelo de regresión lineal univariado.
"""
import numpy as np
import pandas as pd


def pregunta_01():
    """
    En este punto se realiza la lectura de conjuntos de datos.
    Complete el código presentado a continuación.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('gm_2008_region.csv',sep=',')

    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
    y = df['life']
    # y=____[____].____
    X = df['fertility']
    # X = ____[____].____

    # Imprima las dimensiones de `y`
    print(y.shape)

    # Imprima las dimensiones de `X`
    print(X.shape)

    # Transforme `y` a un array de numpy usando reshape
    y_reshaped = y.values.reshape(-1,1)

    # Trasforme `X` a un array de numpy usando reshape
    X_reshaped = X.values.reshape(-1,1)

    # Imprima las nuevas dimensiones de `y`
    print(y_reshaped.shape)

    # Imprima las nuevas dimensiones de `X`
    print(X_reshaped.shape)


def pregunta_02():
    """
    En este punto se realiza la impresiÃ³n de algunas estadÃ­sticas bÃ¡sicas
    Complete el cÃ³digo presentado a continuaciÃ³n.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('gm_2008_region.csv',sep=',')
    
    # Imprima las dimensiones del DataFrame
    print(df.shape)
    
    # Imprima la correlaciÃ³n entre las columnas `life` y `fertility` con 4 decimales.
    correl=round(df['life'].corr(df['fertility'],method='pearson'),4)
    print(correl)
    
    # Imprima la media de la columna `life` con 4 decimales.
    media=round(df['life'].mean(),4)
    print(media)
    
    # Imprima el tipo de dato de la columna `fertility`.
    tipo=type(df['life'])
    print(tipo)
    
    # Imprima la correlaciÃ³n entre las columnas `GDP` y `life` con 4 decimales.
    correl2=round(df['GDP'].corr(df['life'],method='pearson'),4)
    print(correl2)


def pregunta_03():
    """
    Entrenamiento del modelo sobre todo el conjunto de datos.
    Complete el cÃ³digo presentado a continuaciÃ³n.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv('gm_2008_region.csv',sep=',')
    
    # Asigne a la variable los valores de la columna `fertility`
    X_fertility=df['fertility'].values.reshape(-1,1)
    
    # Asigne a la variable los valores de la columna `life`
    y_life=df['life'].values.reshape(-1,1)
    
    # Importe LinearRegression
    from sklearn.linear_model import LinearRegression
    
    # Cree una instancia del modelo de regresiÃ³n lineal
    reg = LinearRegression(
        # Ajusta el intercepto?
        fit_intercept=True,
        # Normaliza los datos?
        # Se ignora si fit_intercept=True.
        normalize=False,
    )
    
    # Cree El espacio de predicciÃ³n. Esto es, use linspace para crear
    # un vector con valores entre el mÃ¡ximo y el mÃ­nimo de X_fertility
    prediction_space = np.linspace(X_fertility.min(), X_fertility.max()).reshape(-1,1)
    
    
    # Entrene el modelo usando X_fertility y y_life
    reg.fit(X_fertility,y_life)
    
    # Compute las predicciones para el espacio de predicción
    y_pred = reg.predict(prediction_space)
    
    # Imprima el R^2 del modelo con 4 decimales
    r2=round(reg.score(X_fertility,y_life),4)
    print(r2)


def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el código presentado a continuación.
    """

    # Importe LinearRegression
    # Importe train_test_split
    # Importe mean_squared_error
    from ____ import ____

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = ____

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = ____

    # Asigne a la variable los valores de la columna `life`
    y_life = ____

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) = ____(
        ____,
        ____,
        test_size=____,
        random_state=____,
    )

    # Cree una instancia del modelo de regresión lineal
    linearRegression = ____

    # Entrene el clasificador usando X_train y y_train
    ____.fit(____, ____)

    # Pronostique y_test usando X_test
    y_pred = ____

    # Compute and print R^2 and RMSE
    print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
    rmse = np.sqrt(____(____, ____))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
