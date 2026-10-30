"""Módulo preparar_datos: Genera un .npz por paciente en data/processed a
partir de los registros .edf de CHB-MIT en data/raw.

Los .edf de los 6 pacientes suman ~16 GB y se procesan como .npz para
guardar solo la potencia por canal y la etiqueta de cada ventana. Con estos
datos se entrenan y evalúan los modelos para detección de crisis.
"""

from pathlib import Path

import numpy as np

from anotaciones import REGISTROS
from datos import Dataset

BASE_DIR = Path.cwd()
CARPETA_EDF = BASE_DIR / "data" / "raw"
CARPETA_PROCESADA = BASE_DIR / "data" / "processed"


def preparar_paciente(paciente: str) -> int:
    """Procesa los registros de un paciente y guarda sus ventanas.
    Devuelve la cantidad de ventanas obtenidas.
    """
    dataset = Dataset()
    # Acceso directo a los registros del paciente y a sus crisis.
    registros = REGISTROS[paciente]
    for nombre in registros:
        ruta = CARPETA_EDF / paciente / nombre
        if ruta.exists():
            n = dataset.agregar_registro(ruta, registros[nombre])
            print(f"  {nombre}: {n} ventanas")
        else:
            print(f"  No se encuentra {ruta}.")

    # Guarda Dataset en un .npz propio, de modo que sea posible separar
    # entrenamiento y prueba por paciente.
    etiquetas = dataset.obtener_etiquetas()
    if len(etiquetas) > 0:
        CARPETA_PROCESADA.mkdir(exist_ok=True)
        dataset.guardar(CARPETA_PROCESADA / f"{paciente}.npz")
    crisis = int(np.sum(etiquetas))
    print(f"{paciente}: {len(etiquetas)} ventanas, {crisis} de crisis")
    return len(etiquetas)


def main() -> None:
    """Prepara el dataset de cada paciente y muestra el total de ventanas."""
    total = 0
    for paciente in REGISTROS:
        total = total + preparar_paciente(paciente)
    print(f"Total: {total} ventanas")


main()
