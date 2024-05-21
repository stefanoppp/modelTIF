from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np

class ModelTrainer:
    def __init__(self, model_type='xgboost'):
        if model_type == 'xgboost':
            self.model = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=10, subsample=0.8, colsample_bytree=0.8)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.15, max_depth=7)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=150, max_depth=5)
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        elif model_type == 'svr':
            self.model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        elif model_type == 'knn':
            self.model = KNeighborsRegressor(n_neighbors=8, metric='manhattan')
        elif model_type == 'decision_tree':
            self.model = DecisionTreeRegressor(max_depth=10)
        else:
            raise ValueError("El modelo debe ser 'xgboost', 'gradient_boosting', 'random_forest', 'linear_regression', 'svr', 'knn', o 'decision_tree'")
        
        self.scaler = StandardScaler()

    def split_data(self, X, y, test_size=0.3, validation_size=0.5, random_state=42):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_data(self, X_train, X_val, X_test):
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print("Error cuadrado medio (MSE):", mse)
        print("Coeficiente de determinación (R^2):", r2)
        print("Error absoluto medio (MAE):", mae)
        return mse, r2, mae

    def fit_evaluate(self, X, y):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_data(X_train, X_val, X_test)
        self.train(X_train_scaled, y_train)
        
        print("Evaluación en conjunto de validación:")
        self.evaluate(X_val_scaled, y_val)
        
        print("Evaluación en conjunto de prueba:")
        self.evaluate(X_test_scaled, y_test)

    @staticmethod
    def calculate_aic_bic(X, y, model):
        X = sm.add_constant(X)
        model_with_const = model.fit(X, y)
        y_pred = model_with_const.predict(X)
        n, k = X.shape[0], X.shape[1]
        resid = y - y_pred
        sse = np.sum(resid ** 2)
        
        aic = n * np.log(sse / n) + 2 * k
        bic = n * np.log(sse / n) + k * np.log(n)
        print(aic)
        print(bic)