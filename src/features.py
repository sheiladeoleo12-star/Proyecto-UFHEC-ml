"""
Preparación de variables (features) para el modelo de predicción de retrasos.

Este módulo es compartido por el notebook y por la aplicación web, para que
los datos se transformen exactamente igual al entrenar y al predecir.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Diccionario de variables: nombre original -> nombre en lenguaje de proyectos
# ---------------------------------------------------------------------------
RENOMBRAR = {
    "RemainingDay": "dias_hasta_fecha_limite",
    "ProgressTime": "dias_desde_creacion",
    "workload": "carga_responsable",
    "perofdelay": "pct_atrasos_responsable",
    "no_blockedby": "predecesoras",
    "no_blocking": "tareas_que_bloquea",
    "no_issuelink": "tareas_relacionadas",
    "repetition": "reaperturas",
    "no_comment": "comentarios",
    "no_priority_change": "cambios_prioridad",
    "no_des_change": "cambios_alcance",
    "no_fixversion": "entregas_asociadas",
    "no_fixversion_change": "replanificaciones",
    "discussion": "dias_definicion",
}

DESCRIPCION = {
    "prioridad": "Prioridad de la tarea (1 = trivial ... 5 = bloqueante)",
    "dias_hasta_fecha_limite": "Días que faltan para la fecha límite comprometida",
    "dias_desde_creacion": "Días transcurridos desde que se creó la tarea",
    "carga_responsable": "Tareas abiertas asignadas al responsable",
    "pct_atrasos_responsable": "% histórico de tareas atrasadas del responsable",
    "predecesoras": "Tareas de las que depende (la bloquean)",
    "tareas_que_bloquea": "Tareas que dependen de esta",
    "tareas_relacionadas": "Total de tareas vinculadas",
    "reaperturas": "Veces que la tarea se reabrió (retrabajo)",
    "comentarios": "Comentarios registrados en la tarea",
    "cambios_prioridad": "Veces que se cambió la prioridad",
    "cambios_alcance": "Veces que se cambió la descripción (alcance)",
    "entregas_asociadas": "Entregas o versiones a las que pertenece",
    "replanificaciones": "Veces que se movió a otra entrega",
    "dias_definicion": "Días dedicados a discusión/definición",
}

TIPOS = [
    "Bug",
    "Mejora",
    "Tarea",
    "Subtarea",
    "Nueva funcionalidad",
    "Prueba",
    "Historia de usuario",
    "Documentación",
    "Otro",
]

MAPA_TIPOS = {
    "Bug": "Bug", "Fug": "Bug",
    "Improvement": "Mejora", "Enhancement": "Mejora", "Wish": "Mejora",
    "Enhancement Request": "Mejora",
    "Task": "Tarea", "Code Task": "Tarea",
    "SubTask": "Subtarea", "Sub-task": "Subtarea", "Technical task": "Subtarea",
    "Documentation Sub-Task": "Subtarea",
    "NewFeature": "Nueva funcionalidad", "New Feature": "Nueva funcionalidad",
    "Feature Request": "Nueva funcionalidad",
    "FunctionalTest": "Prueba", "Test": "Prueba", "Release Test": "Prueba",
    "Story": "Historia de usuario",
    "Documentation": "Documentación",
}

MAPA_PRIORIDAD = {
    "Trivial": 1, "Optional": 1,
    "Minor": 2,
    "Major": 3, "To be reviewed": 3,
    "Critical": 4,
    "Blocker": 5,
}

PRIORIDAD_ES = {1: "Trivial", 2: "Baja", 3: "Media", 4: "Alta", 5: "Bloqueante"}

VARIABLES_NUMERICAS = ["prioridad"] + list(RENOMBRAR.values())
COLUMNAS_TIPO = [f"tipo_{t}" for t in TIPOS]
COLUMNAS_MODELO = VARIABLES_NUMERICAS + COLUMNAS_TIPO


# ---------------------------------------------------------------------------
# 2. Carga y limpieza del dataset de entrenamiento
# ---------------------------------------------------------------------------
def cargar_datos(carpeta: str) -> pd.DataFrame:
    """Une los 8 archivos *_due.csv en una sola tabla con la columna 'proyecto'."""
    archivos = sorted(glob.glob(os.path.join(carpeta, "*_due.csv")))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos *_due.csv en {carpeta}")
    partes = [
        pd.read_csv(f).assign(proyecto=os.path.basename(f).split("_")[0])
        for f in archivos
    ]
    return pd.concat(partes, ignore_index=True)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Traduce, homologa categorías, corrige valores imposibles y crea los objetivos."""
    d = df.rename(columns=RENOMBRAR).copy()
    d["fecha_creacion"] = pd.to_datetime(d["openeddate"])
    d["tipo"] = d["type"].map(MAPA_TIPOS).fillna("Otro")
    d["prioridad"] = d["priority"].map(MAPA_PRIORIDAD).fillna(3).astype(int)
    # Un porcentaje no puede superar 100: valores mayores son errores de registro
    d["pct_atrasos_responsable"] = d["pct_atrasos_responsable"].clip(0, 100)
    # Variables objetivo
    d["dias_atraso"] = d["delaydays"]
    d["atrasada"] = (d["delaydays"] > 0).astype(int)
    columnas = (
        ["issuekey", "proyecto", "fecha_creacion", "tipo"]
        + VARIABLES_NUMERICAS
        + ["dias_atraso", "atrasada"]
    )
    return d[columnas]


# ---------------------------------------------------------------------------
# 3. Matriz de entrada del modelo (igual para entrenamiento y predicción)
# ---------------------------------------------------------------------------
def matriz_modelo(d: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la matriz X con las columnas en el orden que espera el modelo."""
    X = d[VARIABLES_NUMERICAS].astype(float).copy()
    tipo = d["tipo"].where(d["tipo"].isin(TIPOS), "Otro")
    for t in TIPOS:
        X[f"tipo_{t}"] = (tipo == t).astype(int).values
    return X[COLUMNAS_MODELO]


def nombre_legible(col: str) -> str:
    """Nombre amigable para gráficos y explicaciones."""
    if col.startswith("tipo_"):
        return f"Tipo: {col[5:]}"
    return col.replace("_", " ").capitalize().replace("Pct", "%")
