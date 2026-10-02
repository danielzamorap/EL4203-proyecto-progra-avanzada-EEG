"""Módulo datos: Contiene la clase Dataset para el procesamiento de señales en
registros .edf. Se procesan ventanas EEG con sus características y
etiquetas.
"""

from pathlib import Path

import mne
import numpy as np

N_CANALES = 23
FRECUENCIA = 256.0
VENTANA_S = 4


class Dataset:
    """Guarda la potencia por canal de la señal (uV^2) y la etiqueta de cada
    ventana para entrenamiento de modelos.
    """

    def __init__(self):
        """Inicializa características y etiquetas. Crea el Dataset vacío."""
        self.__caracteristicas = []
        self.__etiquetas = []

    def agregar_registro(self, ruta: Path, crisis: list) -> int:
        """Procesa las ventanas de un .edf crudo y devuelve cuántas agregó."""
        crudo = mne.io.read_raw_edf(ruta, preload=False, verbose=False)

        # Se pide el mismo montaje (cantidad de canales y frecuencia) que el
        # dataset.
        if len(crudo.ch_names) != N_CANALES:
            print(f"ERROR: Se esperan {N_CANALES} canales.")
            return 0
        if crudo.info["sfreq"] != FRECUENCIA:
            print(f"ERROR: Se espera frecuencia de {FRECUENCIA} Hz.")
            return 0

        # Se valida que los intervalos de crisis sean coherentes.
        duracion = crudo.n_times / FRECUENCIA
        for inicio, fin in crisis:
            if inicio < 0 or fin <= inicio or fin > duracion:
                print("ERROR: Hay un intervalo de crisis inválido.")
                return 0

        # Carga y lee la matriz real.
        senal = crudo.get_data(units="uV")
        muestras = int(VENTANA_S * FRECUENCIA)
        n_ventanas = crudo.n_times // muestras
        for i in range(n_ventanas):
            # Segmenta el registro EEG continuo en ventanas.
            segmento = senal[:, i * muestras : (i + 1) * muestras]
            fila = np.zeros(N_CANALES)
            # Calcula la potencia media por canal (uV^2).
            for canal in range(N_CANALES):
                fila[canal] = np.mean(segmento[canal, :] ** 2)

            inicio_ventana = i * VENTANA_S
            fin_ventana = inicio_ventana + VENTANA_S
            etiqueta = 0
            # Asigna etiqueta: 1 si la ventana coincide con una crisis.
            for inicio, fin in crisis:
                if inicio_ventana < fin and fin_ventana > inicio:
                    etiqueta = 1

            self.__caracteristicas.append(fila)
            self.__etiquetas.append(etiqueta)
        return n_ventanas

    def obtener_caracteristicas(self) -> np.ndarray:
        """Devuelve una matriz nueva: Cantidad de ventanas (filas) x canales
        (columnas), de valores la potencia media.
        """
        if len(self.__caracteristicas) == 0:
            return np.zeros((0, N_CANALES))
        return np.array(self.__caracteristicas)

    def obtener_etiquetas(self) -> np.ndarray:
        """Devuelve un vector nuevo con las etiquetas (0 normal, 1 crisis)."""
        return np.array(self.__etiquetas)

    def guardar(self, ruta: Path) -> None:
        """Guarda características y etiquetas en un archivo .npz."""
        # Se escoge float32 sobre float64 porque ocupa la mitad de memoria y
        # su resolución es suficiente para la potencia usada.
        np.savez_compressed(
            ruta,
            caracteristicas=np.array(self.__caracteristicas, dtype=np.float32),
            etiquetas=np.array(self.__etiquetas),
        )

    def cargar(self, ruta: Path) -> int:
        """Reemplaza los datos con los del dataset preprocesado .npz y
        retorna cuántas ventanas hay.
        """

        # Verifica que el archivo exista en la ruta.
        if not Path(ruta).exists():
            print(f"ERROR: No existe el archivo {ruta}.")
            return 0

        # Abre el archivo .npz y lee su contenido.
        datos = np.load(ruta)
        claves = datos.files

        # Comprueba que el archivo contenga las matrices que fueron guardadas
        # previamente.
        if "caracteristicas" not in claves or "etiquetas" not in claves:
            datos.close()
            print("ERROR: El archivo no tiene el formato esperado.")
            return 0

        # Saca las matrices a la RAM.
        caracteristicas = datos["caracteristicas"]
        etiquetas = datos["etiquetas"]
        datos.close()

        # Se valida formato de matriz (ventanas x canales) y etiqueta por
        # ventana uno a uno.
        if (
            len(caracteristicas.shape) != 2
            or caracteristicas.shape[1] != N_CANALES
            or len(caracteristicas) != len(etiquetas)
        ):
            print("ERROR: El archivo no tiene el formato esperado.")
            return 0

        # Convierte matrices y vectores a listas para poder agregar más
        # registros y se retorna el total de ventanas cargadas.
        self.__caracteristicas = list(caracteristicas)
        self.__etiquetas = list(etiquetas)
        return len(self.__etiquetas)
