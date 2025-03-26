import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

total = 100
accuracy = (40 + 30) / total
precision = 40 / (40 + 20)
recall = 40 / (40 + 10)
f1 = 2 * (precision * recall) / (precision + recall)

print("Métricas calculadas manualmente:")
print(f"Exactitud: {accuracy:.3f}")
print(f"Precisión: {precision:.3f}")
print(f"Medida-F1: {f1:.3f}")

y_true = (
    [1] * 40 +
    [0] * 30 +
    [0] * 20 +
    [1] * 10
)

y_pred = (
    [1] * 40 +
    [0] * 30 +
    [1] * 20 +
    [0] * 10
)

print("\nMétricas usando scikit-learn:")
print(f"Exactitud: {accuracy_score(y_true, y_pred):.3f}")
print(f"Precisión: {precision_score(y_true, y_pred):.3f}")
print(f"Medida-F1: {f1_score(y_true, y_pred):.3f}")

matriz_confusion = confusion_matrix(y_true, y_pred)
print("\nMatriz de confusión:")
print(matriz_confusion)