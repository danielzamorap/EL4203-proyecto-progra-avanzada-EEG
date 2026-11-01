# Clasificador de crisis epilépticas en EEG

Proyecto del curso EL4203 - Programación Avanzada, Universidad de Chile.

**Integrante:** Daniel Zamora

**Profesor:** Jonas Peñailillo P.

**Auxiliar:** Lucas Marshall R.

**Ayudantes:** Felipe Alarcón V., Felipe Muñoz T.

## Requisitos

Python 3.10 o superior.

## Ejecución

Desde terminal:

```
git clone https://github.com/danielzamorap/EL4203-proyecto-progra-avanzada-EEG.git
cd EL4203-proyecto-progra-avanzada-EEG
python -m pip install -r requirements.txt
python src/main.py
```

El código usa como dataset los archivos `.npz` desde `data/processed/`, generados a partir de los registros `.edf` y los archivos `summary` de 6 pacientes del dataset clínico CHB-MIT. Dado que los `.npz` guardan la potencia por canal y la etiqueta de cada ventana, no es necesario descargar el dataset original para ejecutar el código.