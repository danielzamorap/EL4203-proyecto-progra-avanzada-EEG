"""Módulo ordenamiento: Implementa el algoritmo Merge Sort, usado para
ordenar las ventanas según su puntaje de crisis.
"""


def merge_sort(arreglo: list) -> list:
    """Ordena de menor a mayor en una lista nueva, sin cambiar la original."""
    # Caso base: Una lista de 0 o 1 elementos ya está ordenada.
    if len(arreglo) <= 1:
        return arreglo.copy()
    # Divide la lista en dos mitades y ordena cada una por separado.
    mitad = len(arreglo) // 2
    izquierda = merge_sort(arreglo[:mitad])
    derecha = merge_sort(arreglo[mitad:])
    return mezclar(izquierda, derecha)


def mezclar(izquierda: list, derecha: list) -> list:
    """Une dos listas ordenadas en una nueva lista ordenada."""
    resultado = []
    # Se inician las posiciones de las listas en 0, i para izquierda y j para
    # derecha. Luego se compara el primer elemento pendiente de cada lista y
    # se agrega el de menor valor.
    i = 0
    j = 0
    while i < len(izquierda) and j < len(derecha):
        if izquierda[i] <= derecha[j]:
            resultado.append(izquierda[i])
            i = i + 1
        else:
            resultado.append(derecha[j])
            j = j + 1
    # Cuando ya se agregaron todos los elementos de una lista, se añaden los
    # elementos pendientes de la lista restante.
    return resultado + izquierda[i:] + derecha[j:]
