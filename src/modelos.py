"""Módulo modelos: En este archivo se definen las clases para los modelos de
detección de crisis. Las clases ModeloUmbral y ModeloRedNeuronal heredan de
la clase Modelo.
"""

import numpy as np

from datos import N_CANALES, Dataset


class Modelo:
    """Clase padre de los modelos predictivos: Para toda clase hija define
    la misma estructura base que heredan para entrenar y predecir.
    """

    def __init__(self):
        """Inicializa el modelo sin entrenar."""
        self.__entrenado = False

    def esta_entrenado(self) -> bool:
        """Indica si el modelo ya fue entrenado."""
        return self.__entrenado

    def entrenar(self, dataset: Dataset) -> bool:
        """Comprueba una estructura valida del dataset con ambas etiquetas
        (0: Normal y 1: Crisis). Ajusta los parámetros y registra el estado
        final del modelo entrenado.
        """
        x = dataset.obtener_caracteristicas()
        y = dataset.obtener_etiquetas()

        # Verifica que haya al menos una muestra normal y una de crisis.
        if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
            print("ERROR: El dataset debe tener ventanas de ambas clases.")
            return False

        # Ejecuta el método de aprendizaje del modelo hijo y actualiza estado.
        if self.aprender(x, y):
            self.__entrenado = True
        else:
            self.__entrenado = False
        return self.__entrenado

    def aprender(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Ajusta el modelo a los datos segun lo que se define en cada clase
        hija.
        """
        return False

    def calcular_puntaje(self, x: np.ndarray) -> np.ndarray:
        """Devuelve un puntaje de crisis por segmento EEG definido en cada
        hijo.
        """
        return None

    def obtener_umbral(self) -> float:
        """Devuelve el límite de decisión: Puntaje mínimo para que una
        ventana de la señal sea etiquetada como crisis.
        """
        return 0.5

    def predecir(self, x: np.ndarray) -> np.ndarray:
        """Devuelve un vector de predicciones con 1 o 0 (crisis o normal)
        para cada ventana de entrada.
        """
        # Verifica que el modelo haya sido entrenado previamente.
        if not self.__entrenado:
            print("ERROR: El modelo no ha sido entrenado.")
            return None
        x = np.array(x)

        # Comprueba el formato de la matriz de entrada (ventanas x canales).
        if len(x.shape) != 2 or x.shape[1] != N_CANALES:
            print(f"ERROR: Se esperan {N_CANALES} valores por ventana.")
            return None

        # Calcula los puntajes de crisis utilizando el método del modelo
        # específico.
        puntajes = self.calcular_puntaje(x)
        if puntajes is None:
            return None

        # Genera el vector de predicción aplicando el umbral a los puntajes
        # obtenidos.
        prediccion = np.zeros(len(puntajes))
        prediccion[puntajes >= self.obtener_umbral()] = 1
        return prediccion


class ModeloUmbral(Modelo):
    """Modelo baseline: Define clasificación de crisis si la potencia media
    de los canales de ventana EEG superan un umbral aprendido.
    """

    def __init__(self):
        """Inicializa el modelo con umbral en cero."""
        super().__init__()
        self.__umbral = 0.0

    def obtener_umbral(self) -> float:
        """Devuelve el valor del umbral aprendido (uV^2)."""
        return self.__umbral

    def aprender(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Fija el umbral entre la potencia media ambas etiquetas."""
        # Calcula el puntaje (potencia media) de todas las ventanas de
        # entrenamiento.
        potencia = self.calcular_puntaje(x)

        # Calcula la potencia media para las ventanas de etiqueta normal o
        # crisis.
        normal = np.mean(potencia[y == 0])
        crisis = np.mean(potencia[y == 1])

        # Establece el umbral en el valor medio entre los promedios de ambas
        # etiquetas.
        self.__umbral = (normal + crisis) / 2
        return True

    def calcular_puntaje(self, x: np.ndarray) -> np.ndarray:
        """Devuelve la potencia media de los canales en cada ventana."""
        # Suma la potencia de todos los canales de cada ventana.
        potencia = np.zeros(len(x))
        for canal in range(N_CANALES):
            potencia = potencia + x[:, canal]
        # Retorna el promedio dividiendo la suma por la cantidad de canales.
        return potencia / N_CANALES


class ModeloRedNeuronal(Modelo):
    """Modelo de Red neuronal que sigue una arquitectura MLP. Clasifica
    ventanas a partir de la potencia.

    Se busca implementar con PyTorch y usar GPU si está disponible.
    """

    def __init__(
        self, capas: list | None = None, tasa: float = 1e-3, epocas: int = 50
    ):
        """Inicializa la red neuronal llamando a la clase base (super) y
        estableciendo sus hiperparámetros.
        """
        super().__init__()

        # Valores por defecto para los hiperparámetros.
        self.__capas = [25]
        self.__tasa = 1e-3
        self.__epocas = 50
        if capas is None:
            capas = [25]

        # Valida y ajusta los hiperparámetros ingresados.
        self.ajustar_parametros(capas, tasa, epocas)

    def obtener_parametros(self) -> dict:
        """Devuelve un diccionario con los hiperparámetros actuales de la
        red: La lista de capas, la tasa de aprendizaje y el número de
        épocas.
        """
        return {
            "capas": self.__capas.copy(),
            "tasa": self.__tasa,
            "epocas": self.__epocas,
        }

    def ajustar_parametros(
        self, capas: list, tasa: float, epocas: int
    ) -> None:
        """Valida que los hiperparámetros proporcionados sean correctos y
        los guarda.
        """
        if len(capas) == 0 or min(capas) < 1:
            print("ERROR: Las capas deben tener tamaño >= 1.")
        elif tasa <= 0:
            print("ERROR: La tasa de aprendizaje debe ser positiva.")
        elif epocas < 1:
            print("ERROR: Las épocas deben ser al menos 1.")
        # Si todos los parámetros son válidos, se actualizan los atributos de
        # la clase.
        else:
            self.__capas = list(capas)
            self.__tasa = float(tasa)
            self.__epocas = int(epocas)

    def aprender(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Entrena los pesos de la red neuronal utilizando los datos
        ingresados y los hiperparámetros guardados.
        """

    def calcular_puntaje(self, x: np.ndarray) -> np.ndarray:
        """Devuelve la probabilidad (calculada por la red) de que una
        ventana corresponda a una crisis (salida en el rango [0, 1])
        """
