import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X = np.array([
    [2, 0],  
    [4, 4],  
    [1, 1],  
    [2, 4],  
    [2, 2],  
    [2, 3],  
    [3, 4],  
    [3, 3]   
])

y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

knn = KNeighborsClassifier(n_neighbors=1, p=1)
knn.fit(X, y)

punto_nuevo = np.array([[2.5, 2.5]])

prediccion = knn.predict(punto_nuevo)

distancias, indices = knn.kneighbors(punto_nuevo)

print(f"Punto a clasificar: (2.5, 2.5)")
print(f"Clase predicha: {prediccion[0]}")
print(f"Distancia al vecino más cercano: {distancias[0][0]}")
print(f"Índice del vecino más cercano: {indices[0][0]}")
print(f"Coordenadas del vecino más cercano: {X[indices[0][0]]}")