"""Modulo metricas: Presenta la clase Metricas, cuyo objetivo es evaluar el
rendimiento de un modelo de aprendizaje automático vs los eventos de crisis
anotados en el dataset.
"""

import numpy as np

from datos import Dataset
from modelos import Modelo


class Metricas:
    """Calcula la matriz de confusión y otras métricas de desempeño
    evaluadas sobre los resultados del modelo.
    """

    def __init__(self, modelo: Modelo, dataset: Dataset):
        """Guarda el modelo ya entrenado y el dataset para su validación."""
        self.__modelo = modelo
        self.__dataset = dataset

    def matriz_confusion(self) -> np.ndarray:
        """Devuelve la matriz de confusión [[VN, FP], [FN, VP]]."""
        # Extrae el vector de etiquetas y la matriz de características
        # (potencia media) del dataset.
        y_real = self.__dataset.obtener_etiquetas()
        x = self.__dataset.obtener_caracteristicas()
        # El modelo evalúa las características y genera un vector de etiquetas
        # predichas.
        y_pred = self.__modelo.predecir(x)
        if y_pred is None:
            print("ERROR: No se pudo predecir con el modelo.")
            return None
        # Componentes matriz de confusión:
        vn = np.sum(y_pred[y_real == 0] == 0)
        fp = np.sum(y_pred[y_real == 0] == 1)
        fn = np.sum(y_pred[y_real == 1] == 0)
        vp = np.sum(y_pred[y_real == 1] == 1)
        return np.array([[vn, fp], [fn, vp]])

    def resumen(self) -> dict:
        """Calcula y devuelve un diccionario con 5 métricas que cuantifican
        el desempeño del modelo: Sensitivity, specificity, accuracy,
        precision y F1 score.
        """
        matriz = self.matriz_confusion()
        if matriz is None or np.sum(matriz) == 0:
            return None
        vn, fp = matriz[0, 0], matriz[0, 1]
        fn, vp = matriz[1, 0], matriz[1, 1]
        # Sensitivity.
        sensibilidad = 0.0
        if vp + fn > 0:
            sensibilidad = vp / (vp + fn)
        # Specificity.
        especificidad = 0.0
        if vn + fp > 0:
            especificidad = vn / (vn + fp)
        # Accuracy.
        exactitud = (vn + vp) / np.sum(matriz)
        # Precision.
        precision = 0.0
        if vp + fp > 0:
            precision = vp / (vp + fp)
        # F1 Score.
        f1 = 0.0
        if precision + sensibilidad > 0:
            f1 = 2 * precision * sensibilidad / (precision + sensibilidad)
        return {
            "sensibilidad": float(sensibilidad),
            "especificidad": float(especificidad),
            "exactitud": float(exactitud),
            "precision": float(precision),
            "f1": float(f1),
        }
