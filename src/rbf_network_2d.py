import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class RBFNetwork2D:
    def __init__(self, goal=0.0371, spread=None):
        self.goal = goal
        self.spread = spread
        self.centers = None
        self.weights = None
        self.errors = []
        self.num_neurons = 0
        self.scaler = StandardScaler()
        
    def rbf(self, X, C):
        """Радиальная базисная функция (Гауссиана)"""
        dist = np.sum((X - C) ** 2, axis=1)
        return np.exp(-dist / (2 * self.spread ** 2))
    
    def calculate_activations(self, X):
        """Вычисление активаций для всех центров"""
        activations = np.zeros((X.shape[0], len(self.centers)))
        for i, center in enumerate(self.centers):
            activations[:, i] = self.rbf(X, center)
        return activations
    
    def fit(self, X, y, max_neurons=50):
        """Обучение сети"""
        X = self.scaler.fit_transform(X)
        self.original_X = X.copy()  # Сохраняем для визуализации
        self.original_y = y.copy()
        
        # Инициализация параметра spread
        if self.spread is None:
            dists = []
            for i in range(min(1000, len(X))):
                for j in range(i + 1, min(1000, len(X))):
                    dists.append(np.linalg.norm(X[i] - X[j]))
            self.spread = np.mean(dists) / np.sqrt(2 * len(X))
        
        # Инициализация центров с помощью K-means
        n_centers = min(max_neurons, len(X))
        kmeans = KMeans(n_clusters=n_centers, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        
        # Вычисление активаций
        activations = self.calculate_activations(X)
        
        # Вычисление весов с регуляризацией
        reg_param = 1e-6
        A = activations.T @ activations + reg_param * np.eye(activations.shape[1])
        b = activations.T @ y
        self.weights = np.linalg.solve(A, b)
        
        # Вычисление ошибки
        y_pred = activations @ self.weights
        mse = np.mean((y - y_pred) ** 2)
        self.errors = [mse]
        self.num_neurons = n_centers
        
        print(f"Обучение завершено:")
        print(f"Количество нейронов: {self.num_neurons}")
        print(f"Итоговая ошибка: {mse:.6f}")
    
    def predict(self, X):
        """Предсказание значений"""
        X = self.scaler.transform(X)
        activations = self.calculate_activations(X)
        return activations @ self.weights

def generate_data(x1=-1.0, x2=1.0, y1=-1.5, y2=1.5, nx=7, ny=9):
    """Генерация данных для обучения"""
    x = np.linspace(x1, x2, nx)
    y = np.linspace(y1, y2, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X**2) * np.exp(-Y**2)
    
    XX = X.flatten()
    YY = Y.flatten()
    ZZ = Z.flatten()
    
    return X, Y, Z, np.column_stack([XX, YY]), ZZ

def plot_surface(X, Y, Z, title, subplot=None):
    """Построение поверхности"""
    if subplot is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = subplot
        
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if subplot is None:
        plt.colorbar(surf)
        plt.savefig(f'{title.lower().replace(" ", "_")}.png')
        plt.close()
    return surf

def plot_training_error(errors, goal):
    """Построение графика ошибки обучения"""
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'b-', label='MSE')
    plt.axhline(y=goal, color='r', linestyle='--', label='Goal')
    plt.xlabel('Итерации')
    plt.ylabel('Среднеквадратичная ошибка')
    plt.title('Характеристика точности обучения')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_error.png')
    plt.close()

def plot_centers_and_weights(rbf_net, X, Y):
    """Визуализация центров и весов сети"""
    plt.figure(figsize=(12, 8))
    
    # Построение тепловой карты весов
    centers_x = rbf_net.centers[:, 0]
    centers_y = rbf_net.centers[:, 1]
    
    plt.scatter(centers_x, centers_y, c=rbf_net.weights, 
               cmap='coolwarm', s=100, marker='o', 
               edgecolors='black', linewidth=1,
               label='Центры RBF')
    
    plt.colorbar(label='Веса')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Расположение центров RBF и их веса')
    plt.grid(True)
    plt.savefig('centers_and_weights.png')
    plt.close()

def plot_error_distribution(rbf_net, X, Y, Z):
    """Визуализация распределения ошибки"""
    Z_pred = rbf_net.predict(np.column_stack([X.flatten(), Y.flatten()])).reshape(Z.shape)
    error = np.abs(Z - Z_pred)
    
    plt.figure(figsize=(12, 8))
    plt.contourf(X, Y, error, levels=20, cmap='viridis')
    plt.colorbar(label='Абсолютная ошибка')
    
    # Добавляем центры RBF
    centers_original = rbf_net.scaler.inverse_transform(rbf_net.centers)
    plt.scatter(centers_original[:, 0], centers_original[:, 1], 
               color='red', marker='x', s=100, label='Центры RBF')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Распределение ошибки аппроксимации')
    plt.legend()
    plt.grid(True)
    plt.savefig('error_distribution.png')
    plt.close()

def plot_rbf_responses(rbf_net, X, Y):
    """Визуализация откликов отдельных RBF нейронов"""
    fig = plt.figure(figsize=(15, 10))
    n_neurons = min(6, rbf_net.num_neurons)  # Показываем до 6 нейронов
    
    for i in range(n_neurons):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        center = rbf_net.centers[i]
        weight = rbf_net.weights[i]
        
        # Вычисляем отклик одного нейрона
        X_flat = np.column_stack([X.flatten(), Y.flatten()])
        response = rbf_net.rbf(rbf_net.scaler.transform(X_flat), center) * weight
        Z_response = response.reshape(X.shape)
        
        surf = ax.plot_surface(X, Y, Z_response, cmap='viridis')
        ax.set_title(f'Нейрон {i+1}\nВес: {weight:.3f}')
        plt.colorbar(surf, ax=ax)
    
    plt.tight_layout()
    plt.savefig('rbf_responses.png')
    plt.close()

def plot_comparison(X, Y, Z, Z_pred):
    """Сравнение исходной функции и аппроксимации"""
    fig = plt.figure(figsize=(15, 6))
    
    # Исходная функция
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = plot_surface(X, Y, Z, 'Исходная функция', ax1)
    plt.colorbar(surf1, ax=ax1)
    
    # Аппроксимация
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = plot_surface(X, Y, Z_pred, 'Аппроксимация', ax2)
    plt.colorbar(surf2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('comparison.png')
    plt.close()

def main():
    # Генерация данных
    print("Генерация данных...")
    X, Y, Z, train_X, train_y = generate_data()
    
    # Построение исходной функции
    print("Построение исходной функции...")
    plot_surface(X, Y, Z, 'Исходная функция')
    
    # Создание и обучение сети
    print("\nОбучение RBF сети...")
    rbf = RBFNetwork2D(goal=0.0371)
    rbf.fit(train_X, train_y)
    
    # Построение графика ошибки обучения
    print("\nПостроение графиков...")
    plot_training_error(rbf.errors, rbf.goal)
    
    # Получение предсказаний
    Z_pred = rbf.predict(train_X).reshape(Y.shape)
    
    # Дополнительные визуализации
    plot_centers_and_weights(rbf, X, Y)
    plot_error_distribution(rbf, X, Y, Z)
    plot_rbf_responses(rbf, X, Y)
    plot_comparison(X, Y, Z, Z_pred)
    
    # Вычисление итоговой ошибки
    mse = np.mean((Z.flatten() - Z_pred.flatten()) ** 2)
    print(f"\nИтоговая среднеквадратичная ошибка: {mse:.6f}")

if __name__ == "__main__":
    main()
