import pymysql
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from training import ModelTrainer
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

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

def training(X, variable):
    modelo = ModelTrainer("decision_tree")
    modelo.fit_evaluate(X, variable)
    
    y_pred = modelo.predict(modelo.scaler.transform(X))
    print("Predicciones:", y_pred)

    mse, r2, mae = modelo.evaluate(modelo.scaler.transform(X), variable)
    print("Evaluación en todo el conjunto de datos:")
    print(f"MSE: {mse}, R2: {r2}, MAE: {mae}")
    ModelTrainer.calculate_aic_bic(modelo.scaler.transform(X), variable, modelo.model)
    print("\n")
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

    modelos=[]

    # Modelo de aceitunas
    for variable in y_aceitunas, y_inflacion, y_precio:
        modelos.append(training(X,variable))

    print("A")

    #  Solicitar una fecha al usuario y predecir
    while True:
        user_input = input("Introduce una fecha (YYYY-MM): ")
        user_date = datetime.strptime(user_input, '%Y-%m')
        user_year = user_date.year
        user_month = user_date.month
        user_X = np.array([[user_year, user_month]])

        try:
            for modelo in modelos:
                user_X_scaled = modelo.scaler.transform(user_X)
                user_prediction = modelo.predict(user_X_scaled)
                print(f"Primera prediccion para {user_input}: {user_prediction[0]}")
        except ValueError:
            print("Fecha no válida. Introduzca la fecha en el formato YYYY-MM-DD.")
    
if __name__ == "__main__":
    main()