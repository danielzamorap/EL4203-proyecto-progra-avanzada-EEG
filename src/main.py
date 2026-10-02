"""Módulo main: Script para validar el funcionamiento de las clases de
análisis EEG. Construye un dataset de prueba usando una muestra de los
registros y evalúa el modelo umbral.
"""

from pathlib import Path

from datos import Dataset
from metricas import Metricas
from modelos import ModeloUmbral

BASE_DIR = Path.cwd()
CARPETA_EDF = BASE_DIR / "data" / "raw" / "chb10"
CARPETA_PROCESADA = BASE_DIR / "data" / "processed"
GUARDADO = CARPETA_PROCESADA / "demo.npz"
REGISTROS = [("chb10_01.edf", []), ("chb10_12.edf", [(6313, 6348)])]


def armar_dataset() -> Dataset:
    """Carga el dataset guardado o lo arma leyendo los .edf."""

    # Inicializa un objeto Dataset vacío.
    dataset = Dataset()

    # Verifica si ya existe el dataset preprocesado para cargarlo directamente
    # a la memoria.
    if GUARDADO.exists() and dataset.cargar(GUARDADO) > 0:
        print(
            f"Ventanas cargadas desde {GUARDADO.name}:",
            len(dataset.obtener_etiquetas()),
        )
        return dataset

    # Itera sobre la lista de registros para procesar cada archivo .edf y
    # extraer sus ventanas según las crisis anotadas.
    for nombre, crisis in REGISTROS:
        ruta = CARPETA_EDF / nombre
        # Comprueba que el archivo crudo exista en la ruta indicada antes de
        # comenzar a extraer características.
        if ruta.exists():
            n = dataset.agregar_registro(ruta, crisis)
            print(f"{nombre}: {n} ventanas")
        else:
            print(f"No se encuentra {ruta}. Revisar la descarga.")

    # Si se agregaron ventanas exitosamente al dataset, crea un directorio para
    # guardar la matriz y el vector en formato .npz.
    if len(dataset.obtener_etiquetas()) > 0:
        CARPETA_PROCESADA.mkdir(exist_ok=True)
        dataset.guardar(GUARDADO)
    return dataset


def main() -> None:
    """Arma el dataset, entrena el modelo umbral y muestra sus métricas."""

    # Prepara el dataset con la matriz de características (potencia media) y el
    # vector de etiquetas.
    dataset = armar_dataset()

    # Comprueba que el dataset contenga ventanas procesadas antes de intentar
    # el entrenamiento y evaluación.
    if len(dataset.obtener_etiquetas()) == 0:
        print("No hay datos para evaluar.")
        return

    # Inicializa el modelo baseline de umbral.
    modelo = ModeloUmbral()

    # Entrena el modelo ajustando su umbral interno con los datos extraídos en
    # el dataset.
    if not modelo.entrenar(dataset):
        return
    print(f"Umbral aprendido: {modelo.obtener_umbral():.1f} uV^2")

    # Inicializa un objeto de la clase Metricas para comparar las predicciones
    # del modelo entrenado contra las etiquetas reales.
    metricas = Metricas(modelo, dataset)

    # Calcula e imprime la matriz de confusión.
    print("Matriz de confusión [[VN, FP], [FN, VP]]:")
    print(metricas.matriz_confusion())

    # Crea e imprime un resumen con las métricas de desempeño.
    for nombre, valor in metricas.resumen().items():
        print(f"{nombre}: {valor:.3f}")
    print("Nota: Se entrena y evalúa con los mismos datos.")


main()
