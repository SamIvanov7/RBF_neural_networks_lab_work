import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class RBFNetwork:
    def __init__(self, num_centers, sigma=1.0):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None
        self.scaler = StandardScaler()
        
    def rbf(self, x, c):
        """Радиальная базисная функция (Гауссиана)"""
        return np.exp(-np.linalg.norm(x - c)**2 / (2 * self.sigma**2))
    
    def fit(self, X, y, epochs=100, learning_rate=0.1):
        """Обучение сети"""
        # Нормализация входных данных
        X = self.scaler.fit_transform(X)
        
        # Определение центров с помощью K-means
        kmeans = KMeans(n_clusters=self.num_centers)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        
        # Инициализация весов
        self.weights = np.random.randn(self.num_centers)
        
        # Обучение
        for _ in range(epochs):
            for i in range(len(X)):
                # Вычисление активаций RBF
                activations = np.array([self.rbf(X[i], c) for c in self.centers])
                
                # Вычисление выхода сети
                output = np.dot(activations, self.weights)
                
                # Вычисление ошибки
                error = y[i] - output
                
                # Обновление весов
                self.weights += learning_rate * error * activations
    
    def predict(self, X):
        """Предсказание"""
        X = self.scaler.transform(X)
        predictions = []
        for x in X:
            activations = np.array([self.rbf(x, c) for c in self.centers])
            predictions.append(np.dot(activations, self.weights))
        return np.array(predictions)

def generate_data():
    """Генерация тестовых данных"""
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = np.sin(X).ravel()
    return X, y

def plot_results(X, y, y_pred, title):
    """Визуализация результатов"""
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, 'b-', label='Исходная функция')
    plt.plot(X, y_pred, 'r--', label='Предсказание RBF')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('rbf_results.png')
    plt.close()

def main():
    # Генерация данных
    X, y = generate_data()
    
    # Создание и обучение RBF сети
    rbf = RBFNetwork(num_centers=15, sigma=0.5)
    rbf.fit(X, y, epochs=100, learning_rate=0.1)
    
    # Получение предсказаний
    y_pred = rbf.predict(X)
    
    # Визуализация результатов
    plot_results(X, y, y_pred, 'Аппроксимация функции с помощью RBF сети')
    
    # Вычисление ошибки
    mse = np.mean((y - y_pred)**2)
    print(f'Среднеквадратичная ошибка: {mse:.6f}')

if __name__ == "__main__":
    main()
