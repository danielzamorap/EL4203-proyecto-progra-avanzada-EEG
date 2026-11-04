"""Módulo modelos: En este archivo se definen las clases para los modelos de
detección de crisis. Las clases ModeloUmbral y ModeloRedNeuronal heredan de
la clase Modelo.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from datos import N_CANALES, Dataset
from ordenamiento import merge_sort

# Semilla fija para resultados reproducibles al entrenar la red neuronal.
SEMILLA = 4203
# Batch size para entrenamiento del MLP.
TAMANO_LOTE = 1024

# Especificidad mínima que define el umbral de decisión. Se usa 0.99 (o bien,
# 1% de falsas alarmas) porque solo el 0.19% de las ventanas son crisis y un
# valor más chico haría que las falsas alarmas superen en mayor medida a las
# crisis reales anotadas en cada summary del dataset crudo.
ESPECIFICIDAD_MINIMA = 0.99


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
        """Ajusta el modelo a los datos según lo que define cada clase hija.

        Retorna True si queda entrenado.
        """
        return False

    def calcular_puntaje(self, x: np.ndarray) -> np.ndarray:
        """Devuelve un puntaje de crisis por segmento EEG, definido en cada
        clase hija.
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
        x = self.__validar_entrada(x)
        if x is None:
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

    def priorizar(self, x: np.ndarray) -> list:
        """Ordena los índices de las ventanas de mayor a menor puntaje, lo que
        permite revisar primero las ventanas más probables de ser crisis.
        """
        x = self.__validar_entrada(x)
        if x is None:
            return None
        puntajes = self.calcular_puntaje(x)
        if puntajes is None:
            return None

        # Ordena los pares (puntaje, índice) usando Merge Sort. El índice
        # permite saber a qué ventana corresponde cada puntaje ordenado.
        pares = []
        for i in range(len(puntajes)):
            pares.append((puntajes[i], i))
        pares = merge_sort(pares)

        # Merge Sort deja los pares de menor a mayor, así que se recorren de
        # atrás hacia adelante y se guarda solo el índice.
        orden = []
        for i in range(len(pares) - 1, -1, -1):
            orden.append(pares[i][1])
        return orden

    def __validar_entrada(self, x: np.ndarray) -> np.ndarray:
        """Comprueba que el modelo esté entrenado y que la entrada sea una
        matriz de ventanas x canales. Devuelve la matriz o None si no es
        válida.
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
        return x


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
        """Fija el umbral entre la potencia media de ventanas normales y de
        crisis.
        """
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

    Se implementa con PyTorch y usa la GPU si está disponible.
    """

    def __init__(
        self,
        capas: list | None = None,
        tasa: float = 1e-4,
        epocas: int = 23,
        validacion: Dataset | None = None,
    ):
        """Inicializa la red neuronal llamando a la clase base (super) y
        estableciendo sus hiperparámetros. Si se entrega un dataset de
        validación, al entrenar se mide validation loss en cada época.
        """
        super().__init__()
        self.__validacion = validacion

        # Hiperparámetros elegidos desde notebooks/experimento_mlp.ipynb.
        self.__capas = [25]
        self.__tasa = 1e-4
        self.__epocas = 23
        if capas is None:
            capas = [25]

        # Se definen al entrenar.
        self.__red = None
        self.__media = None
        self.__desviacion = None
        self.__umbral = 0.5
        self.__perdidas = {"train": [], "validacion": []}

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

    def obtener_perdidas(self) -> dict:
        """Devuelve una copia de train y validation loss para cada época del
        último entrenamiento.
        """
        return {
            "train": self.__perdidas["train"].copy(),
            "validacion": self.__perdidas["validacion"].copy(),
        }

    def obtener_umbral(self) -> float:
        """Devuelve el umbral de decisión aprendido sobre la probabilidad
        de crisis.
        """
        return self.__umbral

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
        torch.manual_seed(SEMILLA)

        # Guarda la media y desviación est. de cada canal en escala
        # logarítmica para normalizar de igual manera las ventanas que
        # se evalúen. Se suma 1e-8 a la desviación para no dividir por cero.
        x_log = np.log1p(x)
        self.__media = np.zeros(N_CANALES)
        self.__desviacion = np.zeros(N_CANALES)
        for canal in range(N_CANALES):
            self.__media[canal] = np.mean(x_log[:, canal])
            self.__desviacion[canal] = np.std(x_log[:, canal]) + 1e-8

        entradas = torch.tensor(self.__normalizar(x), dtype=torch.float32)
        etiquetas = torch.tensor(y, dtype=torch.float32)
        lotes = DataLoader(
            TensorDataset(entradas, etiquetas),
            batch_size=TAMANO_LOTE,
            shuffle=True,
        )

        # Al haber muy pocas crisis, el error en cada ventana de crisis se
        # multiplica por 'normales / crisis', para que ambas clases pesen lo
        # mismo en loss.
        device = self.__device()
        peso = torch.tensor([np.sum(y == 0) / np.sum(y == 1)])
        perdida = nn.BCEWithLogitsLoss(pos_weight=peso.to(device))
        red = self.__crear_red().to(device)
        optimizador = torch.optim.Adam(red.parameters(), lr=self.__tasa)

        self.__perdidas = {"train": [], "validacion": []}
        red.train()
        for epoca in range(self.__epocas):
            # Guarda train loss promedio en la época, ponderando cada lote
            # según su tamaño.
            suma = 0.0
            for lote_x, lote_y in lotes:
                optimizador.zero_grad()
                salida = red(lote_x.to(device)).squeeze(1)
                error = perdida(salida, lote_y.to(device))
                error.backward()
                optimizador.step()
                suma = suma + error.item() * len(lote_y)
            self.__perdidas["train"].append(suma / len(y))
            mensaje = (
                f"Epoch {epoca + 1}/{self.__epocas} - "
                f"Train loss: {suma / len(y):.4f}"
            )

            # Si hay datos de validación, mide y muestra validation loss.
            if self.__validacion is not None:
                perdida_val = self.__perdida_validacion(red, perdida)
                self.__perdidas["validacion"].append(perdida_val)
                mensaje = f"{mensaje} - Val loss: {perdida_val:.4f}"
            print(mensaje)
        self.__red = red

        # El peso de las crisis hace que la RN sobreestime su probabilidad, y
        # muchas ventanas normales superan 0.5. Por eso el umbral no es 0.5,
        # sino el que fija ESPECIFICIDAD_MINIMA.
        puntajes = self.calcular_puntaje(x)
        self.__umbral = float(
            np.percentile(puntajes[y == 0], 100 * ESPECIFICIDAD_MINIMA)
        )
        return True

    def calcular_puntaje(self, x: np.ndarray) -> np.ndarray:
        """Devuelve la probabilidad de que cada ventana sea una crisis."""
        if self.__red is None:
            return None
        device = self.__device()
        entradas = torch.tensor(self.__normalizar(x), dtype=torch.float32)
        # Se evalúa la red sin calcular gradientes. La sigmoide convierte su
        # salida en una probabilidad.
        self.__red.eval()
        with torch.no_grad():
            salida = self.__red(entradas.to(device)).squeeze(1)
        return torch.sigmoid(salida).cpu().numpy()

    def __perdida_validacion(
        self, red: nn.Sequential, perdida: nn.BCEWithLogitsLoss
    ) -> float:
        """Calcula validation loss con la misma función de pérdida del
        entrenamiento. No modifica los pesos.
        """
        device = self.__device()
        entradas = torch.tensor(
            self.__normalizar(self.__validacion.obtener_caracteristicas()),
            dtype=torch.float32,
        )
        etiquetas = torch.tensor(
            self.__validacion.obtener_etiquetas(), dtype=torch.float32
        )
        red.eval()
        with torch.no_grad():
            salida = red(entradas.to(device)).squeeze(1)
            valor = perdida(salida, etiquetas.to(device)).item()
        # Vuelve al modo de entrenamiento para seguir con la siguiente época.
        red.train()
        return valor

    def __crear_red(self) -> nn.Sequential:
        """Arma el MLP apilando una capa lineal y una ReLU por cada valor de
        capas, más una neurona de salida.
        """
        red = nn.Sequential()
        entradas = N_CANALES
        for neuronas in self.__capas:
            red.append(nn.Linear(entradas, neuronas))
            red.append(nn.ReLU())
            entradas = neuronas
        red.append(nn.Linear(entradas, 1))
        return red

    def __normalizar(self, x: np.ndarray) -> np.ndarray:
        """Pasa la potencia a escala logarítmica y la estandariza por canal
        con la media y la desviación del entrenamiento.
        """
        x_log = np.log1p(x)
        normalizada = np.zeros(x_log.shape)
        for canal in range(N_CANALES):
            normalizada[:, canal] = (
                x_log[:, canal] - self.__media[canal]
            ) / self.__desviacion[canal]
        return normalizada

    def __device(self) -> str:
        """Elige "cuda" si hay una GPU disponible, "cpu" si no."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
