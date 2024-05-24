import pymysql
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from training import ModelTrainer
import numpy as np

def connect_to_database():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        database='datasettif'
    )
    return conn

def fetch_data_from_db(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT precio, aceitunas, inflacion, fecha FROM datos")
    data = cursor.fetchall()
    return data

def preprocess_data(data):
    aceitunas = []
    inflacion = []
    fechas = []
    precios = [] 

    for fila in data:
        precios.append(fila[0])  
        aceitunas.append(fila[1])
        inflacion.append(fila[2])
        fechas.append(fila[3].strftime('%Y-%m-%d'))
    return fechas, aceitunas, inflacion, precios 

def plot_data(df):
    sns.pairplot(df[['Precio aceite', 'Aceitunas', 'Inflacion']])
    plt.show()

def training(X, variable, model_type):
    modelo = ModelTrainer(model_type)
    modelo.fit_evaluate(X, variable)
    return modelo

def predict(X, variable, modelo):
    y_pred = modelo.predict(modelo.scaler.transform(X))
    print("Predicciones:", y_pred)

    mse, r2, mae = modelo.evaluate(modelo.scaler.transform(X), variable)
    print("Evaluación en todo el conjunto de datos:")
    print(f"MSE: {mse}, R2: {r2}, MAE: {mae}")
    ModelTrainer.calculate_aic_bic(modelo.scaler.transform(X), variable, modelo.model)
    return modelo

def main():
    conn = connect_to_database()
    data = fetch_data_from_db(conn)
    fechas, aceitunas, inflacion, precios = preprocess_data(data)

    years = [datetime.strptime(fecha, '%Y-%m-%d').year for fecha in fechas]
    months = [datetime.strptime(fecha, '%Y-%m-%d').month for fecha in fechas]

    X = np.array(list(zip(years, months)))

    y_aceitunas = np.array(aceitunas)
    y_inflacion=np.array(inflacion)
    y_precio=np.array(precios)

    X_precios=np.array(list(zip(years, months, aceitunas, inflacion)))

    # Modelo de aceitunas
    modelo_aceitunas = training(X, y_aceitunas, model_type="decision_tree")
    modelo_inflacion = training(X, y_inflacion, model_type="random_forest")
    modelo_precio = training(X_precios, y_precio, model_type="gradient_boosting")

    #  Solicitar una fecha al usuario y predecir. 
    while True:
        user_input = input("Introduce una fecha (YYYY-MM): ")
        user_date = datetime.strptime(user_input, '%Y-%m')
        user_year = user_date.year
        user_month = user_date.month
        user_X = np.array([[user_year, user_month]])
        
        aceituna_scaled_input= modelo_aceitunas.scaler.transform(user_X)
        inflacion_scaled_input= modelo_inflacion.scaler.transform(user_X)
        # Almacenamos predicciones de aceituna e inflacion y los usamos como inputs en el ultimo modelo
        prediccion_aceituna = modelo_aceitunas.predict(aceituna_scaled_input)
        prediccion_inflacion = modelo_inflacion.predict(inflacion_scaled_input)
        
        new_X_precios = np.array([[user_year, user_month, prediccion_aceituna[0], prediccion_inflacion[0]]])
        precios_scaled_input=modelo_precio.scaler.transform(new_X_precios)
        prediccion_precio=modelo_precio.predict(precios_scaled_input)

        print("Prediccion de aceitunas procesadas")
        print(prediccion_aceituna)
        print("Prediccion de inflacion")
        print(prediccion_inflacion)
        print("Prediccion de precio")
        print(prediccion_precio)
if __name__ == "__main__":
    main()