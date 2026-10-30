"""Módulo main: Script para validar el funcionamiento de las clases de
análisis EEG. Entrena el modelo umbral con 5 pacientes y lo evalúa con un
paciente nuevo (no usado para entrenamiento).
"""

from pathlib import Path

from datos import cargar_pacientes
from metricas import Metricas
from modelos import ModeloUmbral

CARPETA_PROCESADA = Path.cwd() / "data" / "processed"
PACIENTES = ["chb01", "chb03", "chb04", "chb05", "chb09", "chb10"]
# Paciente que se deja fuera del entrenamiento para evaluar el modelo.
PACIENTE_PRUEBA = "chb10"


def main() -> None:
    """Entrena el modelo umbral y lo evalúa con un paciente no visto."""

    # Separa los pacientes en 5 de entrenamiento y 1 de prueba.
    pacientes_entrenamiento = []
    for paciente in PACIENTES:
        if paciente != PACIENTE_PRUEBA:
            pacientes_entrenamiento.append(paciente)
    entrenamiento = cargar_pacientes(
        CARPETA_PROCESADA, pacientes_entrenamiento
    )
    prueba = cargar_pacientes(CARPETA_PROCESADA, [PACIENTE_PRUEBA])

    # Comprueba que se hayan leído los .npz de todos los pacientes.
    if entrenamiento is None or prueba is None:
        print("No hay datos para entrenar o evaluar.")
        return
    n_entrenamiento = len(entrenamiento.obtener_etiquetas())
    n_prueba = len(prueba.obtener_etiquetas())
    print(
        f"Entrenamiento: {n_entrenamiento} ventanas de "
        f"{len(pacientes_entrenamiento)} pacientes"
    )
    print(f"Prueba: {n_prueba} ventanas de {PACIENTE_PRUEBA} (no visto)")

    # Modelos a entrenar y evaluar, cada uno con su nombre. Actualmente solo
    # ModeloUmbral. En una futura iteración se agregará ModeloRedNeuronal
    # cuando esté implementado.
    modelos = [("Modelo umbral", ModeloUmbral())]

    print()
    print("== Evaluación del modelo (requiere las crisis anotadas) ==")
    for nombre, modelo in modelos:
        if not modelo.entrenar(entrenamiento):
            return

        # Inicializa un objeto de la clase Metricas para comparar las
        # predicciones del modelo entrenado contra las etiquetas reales.
        metricas = Metricas(modelo, prueba)
        print(nombre)
        print("Matriz de confusión [[VN, FP], [FN, VP]]:")
        print(metricas.matriz_confusion())
        for metrica, valor in metricas.resumen().items():
            print(f"{metrica}: {valor:.3f}")


main()
