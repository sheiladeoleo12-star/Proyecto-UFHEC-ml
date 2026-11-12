"""
Lógica del asistente de cronogramas.

1. Lee un cronograma (Excel/CSV) con la plantilla del proyecto.
2. Calcula, para cada tarea abierta, las mismas variables con las que se entrenó el modelo.
3. Aplica los modelos (probabilidad de atraso y días de atraso estimados).
4. Simula el cronograma (Monte Carlo) propagando los atrasos por las dependencias,
   para estimar la probabilidad de cumplir la fecha comprometida y la ruta crítica.
5. Ordena las tareas que conviene atender primero.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .features import COLUMNAS_MODELO, TIPOS, matriz_modelo

COLUMNAS_OBLIGATORIAS = [
    "id", "tarea", "tipo", "prioridad", "responsable",
    "fecha_inicio", "fecha_fin_planificada",
]
COLUMNAS_OPCIONALES = {
    "predecesoras": "",
    "estado": "Pendiente",
    "fecha_creacion": None,
    "fecha_fin_real": None,
    "reaperturas": None,
    "comentarios": None,
    "cambios_prioridad": None,
    "cambios_alcance": None,
    "replanificaciones": None,
    "pct_atrasos_responsable": None,
}

PRIORIDAD_TEXTO = {
    "trivial": 1, "muy baja": 1, "baja": 2, "media": 3, "normal": 3,
    "alta": 4, "critica": 4, "crítica": 4, "bloqueante": 5, "urgente": 5,
}

SINONIMOS_TIPO = {
    "bug": "Bug", "error": "Bug", "incidencia": "Bug", "defecto": "Bug",
    "mejora": "Mejora",
    "tarea": "Tarea", "actividad": "Tarea",
    "subtarea": "Subtarea",
    "nueva funcionalidad": "Nueva funcionalidad", "funcionalidad": "Nueva funcionalidad",
    "desarrollo": "Nueva funcionalidad",
    "prueba": "Prueba", "pruebas": "Prueba", "qa": "Prueba", "test": "Prueba",
    "historia de usuario": "Historia de usuario", "historia": "Historia de usuario",
    "documentación": "Documentación", "documentacion": "Documentación",
}

ESTADOS_CERRADOS = {"completada", "completado", "cerrada", "cerrado", "terminada", "hecho", "done"}


PESO_PREVIO = 5        # tareas "virtuales" para suavizar el historial del responsable
Z_MAX = 1.5            # recorte de escenarios extremos en la simulación (± desviaciones)
ATRASO_MAX = 180       # tope de días de atraso simulados por tarea


class ErrorCronograma(ValueError):
    """Error de formato en el cronograma cargado por el usuario."""


# ---------------------------------------------------------------------------
# Lectura y validación
# ---------------------------------------------------------------------------
def _prioridad(valor) -> int:
    if pd.isna(valor):
        return 3
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return int(min(5, max(1, round(valor))))
    return PRIORIDAD_TEXTO.get(str(valor).strip().lower(), 3)


def _tipo(valor) -> str:
    if pd.isna(valor):
        return "Tarea"
    texto = str(valor).strip()
    if texto in TIPOS:
        return texto
    return SINONIMOS_TIPO.get(texto.lower(), "Otro")


def _fechas(serie: pd.Series) -> pd.Series:
    """Convierte fechas en formato ISO (2027-01-15) o latino (15/01/2027)."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie)
    iso = pd.to_datetime(serie, errors="coerce", format="ISO8601")
    faltan = iso.isna() & serie.notna()
    if faltan.any():
        iso[faltan] = pd.to_datetime(serie[faltan].astype(str), errors="coerce", dayfirst=True, format="mixed")
    return iso


def _lista_predecesoras(valor) -> list[str]:
    if pd.isna(valor) or str(valor).strip() == "":
        return []
    texto = str(valor).replace(";", ",")
    return [p.strip().removesuffix(".0") for p in texto.split(",") if p.strip()]


def preparar_cronograma(crudo: pd.DataFrame) -> pd.DataFrame:
    """Valida columnas, normaliza valores y devuelve el cronograma listo para analizar."""
    df = crudo.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltantes:
        raise ErrorCronograma(
            "Faltan columnas obligatorias: " + ", ".join(faltantes)
            + ". Descarga la plantilla para ver el formato."
        )
    df = df.dropna(subset=["id", "tarea"]).copy()
    for col, defecto in COLUMNAS_OPCIONALES.items():
        if col not in df.columns:
            df[col] = defecto

    df["id"] = df["id"].astype(str).str.strip().str.removesuffix(".0")
    if df["id"].duplicated().any():
        repetidos = df.loc[df["id"].duplicated(), "id"].tolist()
        raise ErrorCronograma(f"Hay IDs repetidos: {repetidos}")

    for col in ["fecha_inicio", "fecha_fin_planificada", "fecha_creacion", "fecha_fin_real"]:
        df[col] = _fechas(df[col])
    if df["fecha_inicio"].isna().any() or df["fecha_fin_planificada"].isna().any():
        raise ErrorCronograma("Todas las tareas necesitan fecha_inicio y fecha_fin_planificada válidas.")
    if (df["fecha_fin_planificada"] < df["fecha_inicio"]).any():
        malas = df.loc[df["fecha_fin_planificada"] < df["fecha_inicio"], "id"].tolist()
        raise ErrorCronograma(f"Estas tareas terminan antes de empezar: {malas}")
    df["fecha_creacion"] = df["fecha_creacion"].fillna(df["fecha_inicio"])

    df["tipo"] = df["tipo"].map(_tipo)
    df["prioridad"] = df["prioridad"].map(_prioridad)
    df["lista_predecesoras"] = df["predecesoras"].map(_lista_predecesoras)
    df["cerrada"] = df["estado"].astype(str).str.strip().str.lower().isin(ESTADOS_CERRADOS)
    for col in ["reaperturas", "comentarios", "cambios_prioridad", "cambios_alcance", "replanificaciones"]:
        # Los vacíos se dejan como NaN: luego se completan con el valor típico del dataset
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    ids = set(df["id"])
    desconocidas = {p for lista in df["lista_predecesoras"] for p in lista if p not in ids}
    if desconocidas:
        raise ErrorCronograma(f"Predecesoras que no existen en el cronograma: {sorted(desconocidas)}")
    orden_topologico(df)  # valida que no haya ciclos
    return df.reset_index(drop=True)


def orden_topologico(df: pd.DataFrame) -> list[str]:
    """Orden en que se pueden calcular las tareas respetando dependencias."""
    preds = dict(zip(df["id"], df["lista_predecesoras"]))
    sucesoras = defaultdict(list)
    grado = {i: len(p) for i, p in preds.items()}
    for i, ps in preds.items():
        for p in ps:
            sucesoras[p].append(i)
    cola = [i for i, g in grado.items() if g == 0]
    orden = []
    while cola:
        actual = cola.pop(0)
        orden.append(actual)
        for s in sucesoras[actual]:
            grado[s] -= 1
            if grado[s] == 0:
                cola.append(s)
    if len(orden) != len(preds):
        raise ErrorCronograma("Las dependencias forman un ciclo (A depende de B y B de A).")
    return orden


# ---------------------------------------------------------------------------
# Variables del modelo calculadas desde el cronograma
# ---------------------------------------------------------------------------
def calcular_variables(
    df: pd.DataFrame,
    fecha_corte: pd.Timestamp,
    pct_defecto: float,
    valores_tipicos: dict | None = None,
) -> pd.DataFrame:
    """
    Traduce el cronograma a las variables con que se entrenó el modelo.

    Los datos de seguimiento que el usuario no registra (comentarios, replanificaciones...)
    se completan con el valor típico (mediana) del dataset de entrenamiento. Poner 0 haría
    que el modelo interprete la tarea como "sin seguimiento", lo que eleva el riesgo.
    """
    d = df.copy()
    tipicos = {"reaperturas": 0, "comentarios": 1, "cambios_prioridad": 0,
               "cambios_alcance": 0, "replanificaciones": 1, "entregas_asociadas": 1,
               "dias_definicion": 0}
    tipicos.update(valores_tipicos or {})
    for col in ["reaperturas", "comentarios", "cambios_prioridad", "cambios_alcance", "replanificaciones"]:
        d[col] = d[col].fillna(tipicos[col])
    abiertas = ~d["cerrada"]
    cerradas_por_id = dict(zip(d["id"], d["cerrada"]))

    sucesoras = defaultdict(int)
    for ps in d["lista_predecesoras"]:
        for p in ps:
            sucesoras[p] += 1

    d["dias_hasta_fecha_limite"] = (d["fecha_fin_planificada"] - fecha_corte).dt.days.clip(lower=0)
    d["dias_desde_creacion"] = (fecha_corte - d["fecha_creacion"]).dt.days.clip(lower=0)
    d["carga_responsable"] = d["responsable"].map(d.loc[abiertas, "responsable"].value_counts()).fillna(0)
    d["predecesoras"] = d["lista_predecesoras"].map(
        lambda ps: sum(not cerradas_por_id[p] for p in ps)
    )
    d["tareas_que_bloquea"] = d["id"].map(sucesoras).fillna(0)
    d["tareas_relacionadas"] = d["lista_predecesoras"].map(len) + d["tareas_que_bloquea"]
    d["entregas_asociadas"] = tipicos["entregas_asociadas"]
    d["dias_definicion"] = tipicos["dias_definicion"]

    # Historial del responsable: 1) dato del usuario; 2) calculado con tareas cerradas
    # del propio cronograma que tengan fecha_fin_real; 3) valor típico del dataset.
    # El cálculo propio se suaviza (promedio bayesiano con PESO_PREVIO tareas "virtuales"
    # con el valor típico) para que una sola tarea atrasada no implique 100 %.
    historial = pd.Series(dtype=float)
    con_real = d[d["cerrada"] & d["fecha_fin_real"].notna()]
    if len(con_real):
        tarde = (con_real["fecha_fin_real"] > con_real["fecha_fin_planificada"]).astype(float)
        g = tarde.groupby(con_real["responsable"])
        historial = (g.sum() + PESO_PREVIO * pct_defecto / 100) / (g.size() + PESO_PREVIO) * 100
    pct_usuario = pd.to_numeric(d["pct_atrasos_responsable"], errors="coerce")
    d["pct_atrasos_responsable"] = (
        pct_usuario.fillna(d["responsable"].map(historial)).fillna(pct_defecto).clip(0, 100)
    )
    return d


def predecir(d: pd.DataFrame, clasificador, regresor) -> pd.DataFrame:
    """Agrega probabilidad de atraso, días estimados y atraso esperado."""
    X = matriz_modelo(d)[COLUMNAS_MODELO]
    d = d.copy()
    d["prob_atraso"] = clasificador.predict_proba(X)[:, 1]
    d["dias_atraso_si_ocurre"] = np.expm1(regresor.predict(X)).clip(min=1)
    d["atraso_esperado"] = d["prob_atraso"] * d["dias_atraso_si_ocurre"]
    d.loc[d["cerrada"], ["prob_atraso", "dias_atraso_si_ocurre", "atraso_esperado"]] = 0.0
    return d


# ---------------------------------------------------------------------------
# Simulación Monte Carlo del cronograma
# ---------------------------------------------------------------------------
def simular(
    d: pd.DataFrame,
    fecha_corte: pd.Timestamp,
    sigma_log: float,
    n: int = 2000,
    semilla: int = 42,
    escala: float = 1.0,
    fecha_compromiso: pd.Timestamp | None = None,
) -> dict:
    """
    Simula n escenarios del proyecto. En cada escenario, cada tarea abierta se
    atrasa con la probabilidad del modelo y, si se atrasa, lo hace un número de
    días alrededor de lo estimado por el regresor (con su error histórico).
    Los atrasos se propagan a las tareas que dependen de ella.

    sigma_log: incertidumbre (desviación en escala logarítmica) de los días de atraso.
    escala:    factor de ajuste al contexto (1.0 = magnitudes del dataset de entrenamiento).
    fecha_compromiso: fecha final comprometida con el cliente/patrocinador. Si no se
               indica, se usa la última fecha planificada del cronograma.
    """
    rng = np.random.default_rng(semilla)
    orden = orden_topologico(d)
    info = d.set_index("id")
    origen = info["fecha_inicio"].min()

    def dia(fecha):
        return (fecha - origen).days

    corte = dia(fecha_corte)
    fin = {}
    ini = {}
    motor = {}  # predecesora que determina el inicio (para la ruta crítica)
    for t in orden:
        fila = info.loc[t]
        inicio_plan = dia(fila["fecha_inicio"])
        duracion = max(1, dia(fila["fecha_fin_planificada"]) - inicio_plan)
        preds = fila["lista_predecesoras"]

        if fila["cerrada"]:
            real = fila["fecha_fin_real"] if pd.notna(fila["fecha_fin_real"]) else fila["fecha_fin_planificada"]
            fin[t] = np.full(n, float(dia(real)))
            ini[t] = np.full(n, float(inicio_plan))
            motor[t] = np.full(n, -1)
            continue

        inicio = np.full(n, float(inicio_plan))
        motor_t = np.full(n, -1)
        if preds:
            fines_preds = np.vstack([fin[p] for p in preds])
            mayor = fines_preds.max(axis=0)
            arg = fines_preds.argmax(axis=0)
            manda = mayor > inicio
            motor_t = np.where(manda, arg, -1)
            inicio = np.maximum(inicio, mayor)

        atrasa = rng.random(n) < fila["prob_atraso"]
        mu = np.log1p(fila["dias_atraso_si_ocurre"])
        z = rng.standard_normal(n).clip(-Z_MAX, Z_MAX)
        dias = (np.expm1(mu + sigma_log * z) * escala).clip(1, ATRASO_MAX)
        termino = inicio + duracion + np.where(atrasa, dias, 0.0)
        # Una tarea abierta no puede terminar antes de la fecha de corte
        fin[t] = np.maximum(termino, corte)
        ini[t] = inicio
        motor[t] = motor_t

    matriz_fin = np.vstack([fin[t] for t in orden])
    fin_proyecto = matriz_fin.max(axis=0)
    ultima = matriz_fin.argmax(axis=0)

    # Índice de criticidad: % de escenarios en que la tarea está en la cadena
    # que determina la fecha final del proyecto.
    pos = {t: i for i, t in enumerate(orden)}
    preds_de = {t: info.loc[t, "lista_predecesoras"] for t in orden}
    conteo = np.zeros(len(orden))
    for s in range(n):
        t = orden[ultima[s]]
        while True:
            conteo[pos[t]] += 1
            m = motor[t][s]
            if m < 0:
                break
            t = preds_de[t][m]
    criticidad = pd.Series(conteo / n, index=orden)

    compromiso = fecha_compromiso if fecha_compromiso is not None else info["fecha_fin_planificada"].max()
    fin_plan = dia(compromiso)
    return {
        "origen": origen,
        "fin_proyecto_dias": fin_proyecto,
        "fin_planificado_dias": fin_plan,
        "prob_cumplir": float((fin_proyecto <= fin_plan).mean()),
        "p50": origen + pd.Timedelta(days=float(np.percentile(fin_proyecto, 50))),
        "p80": origen + pd.Timedelta(days=float(np.percentile(fin_proyecto, 80))),
        "p95": origen + pd.Timedelta(days=float(np.percentile(fin_proyecto, 95))),
        "fin_planificado": compromiso,
        "fin_cronograma": info["fecha_fin_planificada"].max(),
        "criticidad": criticidad,
        "fin_medio_tarea": {t: origen + pd.Timedelta(days=float(np.median(fin[t]))) for t in orden},
        "inicio_medio_tarea": {t: origen + pd.Timedelta(days=float(np.median(ini[t]))) for t in orden},
        "fin_p80_tarea": {t: origen + pd.Timedelta(days=float(np.percentile(fin[t], 80))) for t in orden},
    }


def priorizar(d: pd.DataFrame, criticidad: pd.Series) -> pd.DataFrame:
    """
    Puntaje de atención = atraso esperado x (0.5 + índice de criticidad).
    Prioriza tareas con alto riesgo que además empujan la fecha final.
    """
    p = d[~d["cerrada"]].copy()
    p["criticidad"] = p["id"].map(criticidad).fillna(0)
    p["puntaje_atencion"] = p["atraso_esperado"] * (0.5 + p["criticidad"])
    return p.sort_values("puntaje_atencion", ascending=False)
