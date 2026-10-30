"""Módulo datos: Contiene la clase Dataset para el procesamiento de señales en
registros .edf. Se procesan ventanas EEG con sus características y
etiquetas.
"""

import warnings
from pathlib import Path

import mne
import numpy as np

# El montaje de CHB-MIT repite el canal T8-P8. Al leer cada registro, MNE
# lo renombra como T8-P8-0 y T8-P8-1 y muestra un aviso, el cual se ignora
# con warnings.
warnings.filterwarnings("ignore", message="Channel names are not unique")

N_CANALES = 23
FRECUENCIA = 256.0
VENTANA_S = 4


class Dataset:
    """Guarda la potencia por canal de la señal (uV^2) y la etiqueta de cada
    ventana para entrenamiento de modelos.
    """

    def __init__(self):
        """Inicializa características, etiquetas y el origen de cada
        ventana. Crea el Dataset vacío.
        """
        self.__caracteristicas = []
        self.__etiquetas = []
        self.__registros = []
        self.__segundos = []

    def agregar_registro(self, ruta: Path, crisis: list) -> int:
        """Procesa las ventanas de un .edf crudo y devuelve cuántas agregó."""
        crudo = mne.io.read_raw_edf(ruta, preload=False, verbose=False)

        # Se exigen al menos los 23 canales de EEG y la frecuencia del
        # dataset. Algunos registros traen un canal extra al final (ECG o
        # VNS), que se descarta al leer la señal.
        if len(crudo.ch_names) < N_CANALES:
            print(f"ERROR: Se esperan al menos {N_CANALES} canales.")
            return 0
        if crudo.info["sfreq"] != FRECUENCIA:
            print(f"ERROR: Se espera frecuencia de {FRECUENCIA} Hz.")
            return 0

        # Se valida que los intervalos de crisis sean coherentes, es decir,
        # que estén dentro del registro y que el inicio ocurra antes del fin.
        duracion = crudo.n_times / FRECUENCIA
        for inicio, fin in crisis:
            if inicio < 0 or fin <= inicio or fin > duracion:
                print("ERROR: Hay un intervalo de crisis inválido.")
                return 0

        # Carga y lee la matriz real, únicamente con los 23 canales de EEG.
        senal = crudo.get_data(units="uV")[:N_CANALES, :]
        nombre = Path(ruta).name
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
            # Guarda el registro y segundo en que empieza la ventana.
            self.__registros.append(nombre)
            self.__segundos.append(inicio_ventana)
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

    def obtener_origen(self, ventana: int) -> tuple:
        """Indica el registro y el segundo donde empieza una ventana."""
        if ventana < 0 or ventana >= len(self.__registros):
            print("ERROR: La ventana no existe.")
            return None
        return str(self.__registros[ventana]), int(self.__segundos[ventana])

    def guardar(self, ruta: Path) -> None:
        """Guarda características, etiquetas y origen en un archivo .npz."""
        # Se escoge float32 sobre float64 porque ocupa la mitad de memoria y
        # su resolución es suficiente para la potencia usada.
        np.savez_compressed(
            ruta,
            caracteristicas=np.array(self.__caracteristicas, dtype=np.float32),
            etiquetas=np.array(self.__etiquetas),
            registros=np.array(self.__registros),
            segundos=np.array(self.__segundos),
        )

    def cargar(self, ruta: Path) -> int:
        """Agrega los datos del dataset preprocesado .npz a los que ya tiene
        y retorna cuántas ventanas hay en total.
        """

        # Verifica que el archivo exista en la ruta.
        if not Path(ruta).exists():
            print(f"ERROR: No existe el archivo {ruta}.")
            return 0

        # Abre el archivo .npz y obtiene los nombres de sus matrices.
        datos = np.load(ruta)
        claves = datos.files

        # Comprueba que el archivo contenga las matrices que fueron guardadas
        # previamente.
        if (
            "caracteristicas" not in claves
            or "etiquetas" not in claves
            or "registros" not in claves
            or "segundos" not in claves
        ):
            datos.close()
            print("ERROR: El archivo no tiene el formato esperado.")
            return 0

        # Copia las matrices a la RAM y cierra el archivo.
        caracteristicas = datos["caracteristicas"]
        etiquetas = datos["etiquetas"]
        registros = datos["registros"]
        segundos = datos["segundos"]
        datos.close()

        # Se valida el formato de la matriz (ventanas x canales) y que cada
        # ventana tenga su etiqueta y su origen.
        if (
            len(caracteristicas.shape) != 2
            or caracteristicas.shape[1] != N_CANALES
            or len(caracteristicas) != len(etiquetas)
            or len(registros) != len(etiquetas)
            or len(segundos) != len(etiquetas)
        ):
            print("ERROR: El archivo no tiene el formato esperado.")
            return 0

        # Convierte matrices y vectores a listas y las agrega a las que ya
        # tiene, para poder juntar varios pacientes. Se retorna el total de
        # ventanas.
        self.__caracteristicas = self.__caracteristicas + list(caracteristicas)
        self.__etiquetas = self.__etiquetas + list(etiquetas)
        self.__registros = self.__registros + list(registros)
        self.__segundos = self.__segundos + list(segundos)
        return len(self.__etiquetas)


def cargar_pacientes(carpeta: Path, pacientes: list) -> Dataset:
    """Junta en un solo Dataset las ventanas guardadas de varios pacientes
    (carpeta/chbXX.npz). Devuelve None si alguno no se pudo cargar.
    """
    dataset = Dataset()
    for paciente in pacientes:
        if dataset.cargar(Path(carpeta) / f"{paciente}.npz") == 0:
            return None
    return dataset
