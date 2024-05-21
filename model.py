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

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print("Error cuadrado medio:", mse)
    print("Coeficiente de determinación:", r2)
    print("Error absoluto medio:", mae)

def plot_data(df):
    sns.pairplot(df[['Precio aceite', 'Aceitunas', 'Inflacion']])
    plt.show()

def main():
    conn = connect_to_database()
    data = fetch_data_from_db(conn)
    fechas, aceitunas, inflacion, precios = preprocess_data(data)
    fechas_ordinal = [datetime.strptime(fecha, '%Y-%m-%d').toordinal() for fecha in fechas]
    
    # Lista de tipos de modelo
    modelos = ['xgboost', 'gradient_boosting', 'random_forest', 'linear_regression', 'svr', 'knn', 'decision_tree']


    for modelo in modelos:
        print(f"Entrenando y evaluando modelo ACEITUNAS:{modelo}")
        model_trainer = ModelTrainer(modelo)
        X = np.array(fechas_ordinal).reshape(-1, 1) 
        model_trainer.fit_evaluate(X, aceitunas)
        ModelTrainer.calculate_aic_bic(X, aceitunas, model_trainer.model)



    # KNN para inflacion
    # print(f"Entrenando y evaluando modelo INFLACION: knn")
    # model_trainer = ModelTrainer("knn")
    # X = np.array(fechas_ordinal).reshape(-1, 1) 
    # model_trainer.fit_evaluate(X, inflacion)
    # ModelTrainer.calculate_aic_bic(X, inflacion, model_trainer.model)

    print("\n")

    # Random Forest para precios
    # print(f"Entrenando y evaluando modelo PRECIOS: Random Forest")
    # model_trainer = ModelTrainer("random_forest")
    # X = np.array(fechas_ordinal).reshape(-1, 1) 
    # model_trainer.fit_evaluate(X, precios)
    
    # Calcular AIC y BIC
    # ModelTrainer.calculate_aic_bic(X, precios, model_trainer.model)
    
if __name__ == "__main__":
    main()


