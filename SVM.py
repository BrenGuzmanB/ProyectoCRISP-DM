"""
Created on Sun Dec 10 01:36:36 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""

import numpy as np
import matplotlib.pyplot as plt

class support_vector_machine:
    def __init__(self, learning_rate=0.001, lambda_=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_  # Factor de penalización
        self.epochs = epochs  # num iteraciones
        self.w = None  # weights
        self.b = None  # intercept

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # inicializar en ceros
        self.b = 0  # inicializar en 0

        # Mapear etiquetas de clase {-1, 1}
        y_mapped = np.where(y <= 0, -1, 1)

        for epoch in range(self.epochs): # Épocas para el descenso del gradiente
            self.gradient_descent_step(X, y_mapped)

        return self.w, self.b

    def gradient_descent_step(self, X, y):  # Descenso del gradiente
        for i, Xi in enumerate(X):
            # Condición basada en la diferencia entre la puntuación predicha y el margen
            condition = y[i] * (np.dot(Xi, self.w) - self.b) >= 1
            
            # Actualizar los weights (w) 
            self.w -= self.learning_rate * (2 * self.lambda_ * self.w) if condition else self.learning_rate * (
                    2 * self.lambda_ * self.w - np.dot(Xi, y[i]))
            
            # Actualizar el sesgo (b)
            self.b -= self.learning_rate * y[i] if not condition else 0


    def predict(self, X):
        pred = np.dot(X, self.w) - self.b
        #print(pred)
        result = np.where(pred > 0, 1, 0)  # Mapeo a las clases originales [0,1]
        return result

    def get_hyperplane(self, x, offset): #calcula la posición de un hiperplano en el espacio bidimensional
        #retorna el valor de x correspondiente a la posición en el espacio bidimensional determinada por la ecuación del hiperplano.    
        return (-self.w[0] * x + self.b + offset) / self.w[1]   


    def plot_svm(self, X_train, y_train, X_test, y_test, title='Plot for linear SVM'):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    
        # Puntos de entrenamiento 
        scatter_train = ax.scatter(X_train[:, 0], X_train[:, 1], marker='p', c=y_train, cmap='winter', label='Train', edgecolor='k')
    
        # Puntos de prueba
        scatter_test = ax.scatter(X_test[:, 0], X_test[:, 1], marker='*', c=y_test, cmap='winter', label='Test', edgecolor='k')
        
        # Determinar las coordenadas y dimensiones para trazar el hiperplano y márgenes de decisión
        
        # Máximos y mínimos (límites horizontales).
        x0_1 = np.amin(np.concatenate([X_train[:, 0], X_test[:, 0]]))
        x0_2 = np.amax(np.concatenate([X_train[:, 0], X_test[:, 0]]))
        
        #  Posiciones verticales  utilizando los límites horizontales
        x1_1 = self.get_hyperplane(x0_1, 0)
        x1_2 = self.get_hyperplane(x0_2, 0)
        
        # Posiciones verticales para la línea del margen de decisión inferior. 
        x1_1_m = self.get_hyperplane(x0_1, -1)
        x1_2_m = self.get_hyperplane(x0_2, -1)
    
        # Posiciones verticales para la línea del margen de decisión superior
        x1_1_p = self.get_hyperplane(x0_1, 1)
        x1_2_p = self.get_hyperplane(x0_2, 1)
    
        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], '--', color='gray')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], '--', color='gray')
    
        # mínimo y máximo para la segunda característica
        x1_min = np.amin(np.concatenate([X_train[:, 1], X_test[:, 1]]))
        x1_max = np.amax(np.concatenate([X_train[:, 1], X_test[:, 1]]))
        ax.set_ylim([x1_min - 1, x1_max + 1]) #  límites verticales de la gráfica
    
        plt.title(title)
        plt.legend(handles=[scatter_train.legend_elements()[0][0], scatter_test.legend_elements()[0][0]], labels=['Train', 'Test'])
        
        plt.show()
    


