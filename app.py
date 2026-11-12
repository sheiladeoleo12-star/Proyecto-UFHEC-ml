"""
Asistente inteligente de cronogramas
Predicción de retrasos en tareas de proyectos mediante Machine Learning

Ejecutar localmente:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from src.cronograma import (
    ErrorCronograma, calcular_variables, predecir, preparar_cronograma, priorizar, simular,
)
from src.features import COLUMNAS_MODELO, DESCRIPCION, TIPOS, matriz_modelo, nombre_legible

# ---------------------------------------------------------------------------
# Enlaces del proyecto (actualizar después de publicar)
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/sheiladeoleo12-star/Proyecto-UFHEC-ml"
DOC_URL = REPO_URL + "/blob/main/docs/Documento_Tecnico.pdf"
NOTEBOOK_URL = REPO_URL + "/blob/main/notebooks/01_prediccion_retrasos_cronogramas.ipynb"
VIDEO_URL = REPO_URL + "/blob/main/video/Video_Presentacion_Proyecto.mp4"
AUTORA = "Sheila De Oleo"

RAIZ = Path(__file__).parent
EJEMPLO = RAIZ / "data" / "ejemplo_cronograma.xlsx"
PLANTILLA = RAIZ / "data" / "plantilla_cronograma.xlsx"
FIGURAS = RAIZ / "reports" / "figures"

# Paleta (validada para daltonismo) y colores de estado
AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
ESTADO = {"Alto": "#d03b3b", "Medio": "#fab219", "Bajo": "#0ca30c", "Cerrada": "#c3c2b7"}
ICONO = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢", "Cerrada": "✔️"}
INCERTIDUMBRE = {"Baja": 0.4, "Media (recomendada)": 0.6, "Alta": 0.8, "La del modelo (muy alta)": 1.5}

st.set_page_config(page_title="Asistente inteligente de cronogramas", page_icon="📅", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1250px;}
    .hero {padding: 1.6rem 1.8rem; border-radius: 14px; background: linear-gradient(135deg,#0d366b,#2a78d6); color: #fff;}
    .hero h1 {color:#fff; margin:0 0 .4rem 0; font-size:2.1rem;}
    .hero p {color:#e6efff; font-size:1.08rem; margin:0;}
    .tarjeta {border:1px solid rgba(11,11,11,.10); border-radius:12px; padding:1rem 1.1rem; height:100%; background:#fcfcfb;}
    .tarjeta h4 {margin:.1rem 0 .5rem 0;}
    .paso {font-size:2rem; font-weight:700; color:#2a78d6; line-height:1;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    clf = joblib.load(RAIZ / "models" / "clasificador.joblib")
    reg = joblib.load(RAIZ / "models" / "regresor.joblib")
    meta = json.loads((RAIZ / "models" / "metadata.json").read_text(encoding="utf-8"))
    return clf, reg, meta


CLF, REG, META = cargar_modelos()
UMBRAL = META["umbral"]


def nivel(p: float, cerrada: bool = False) -> str:
    if cerrada:
        return "Cerrada"
    if p >= UMBRAL:
        return "Alto"
    return "Medio" if p >= 0.30 else "Bajo"


def contribuciones(X: pd.DataFrame) -> pd.DataFrame:
    """Contribuciones SHAP calculadas por XGBoost (pred_contribs), en escala log-odds."""
    booster = CLF.get_booster()
    c = booster.predict(xgb.DMatrix(X[COLUMNAS_MODELO]), pred_contribs=True)
    return pd.DataFrame(c[:, :-1], columns=COLUMNAS_MODELO, index=X.index)


def grafico_explicacion(fila_x: pd.Series, fila_c: pd.Series, top: int = 8) -> go.Figure:
    """Barras horizontales: qué sube y qué baja el riesgo de una tarea."""
    c = fila_c.copy()
    # Agrupa el tipo de tarea en una sola barra
    tipo_cols = [k for k in c.index if k.startswith("tipo_")]
    tipo_activo = next((k[5:] for k in tipo_cols if fila_x[k] == 1), "Otro")
    c["tipo"] = c[tipo_cols].sum()
    c = c.drop(tipo_cols)
    c = c.reindex(c.abs().sort_values(ascending=False).index)[:top][::-1]

    def etiqueta(k):
        if k == "tipo":
            return f"Tipo de tarea = {tipo_activo}"
        v = fila_x[k]
        v = f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
        return f"{nombre_legible(k)} = {v}"

    fig = go.Figure(go.Bar(
        x=c.values, y=[etiqueta(k) for k in c.index], orientation="h",
        marker_color=[NARANJA if v > 0 else AZUL for v in c.values],
        hovertemplate="%{y}<br>Efecto: %{x:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=40 + 34 * len(c), margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="← reduce el riesgo · aumenta el riesgo →", plot_bgcolor="#fcfcfb",
        xaxis=dict(zeroline=True, zerolinecolor="#c3c2b7", gridcolor="#e1e0d9"),
    )
    return fig


def a_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        df.to_excel(w, index=False, sheet_name="analisis")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Encabezado y pestañas
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <h1>📅 Asistente inteligente de cronogramas</h1>
      <p>Machine Learning para anticipar qué tareas se van a atrasar, cuánto, y qué hacer primero
      para cumplir la fecha comprometida.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

tab_inicio, tab_cron, tab_tarea, tab_result, tab_acerca = st.tabs(
    ["🏠 Inicio", "📅 Analizar cronograma", "🔍 Evaluar una tarea", "📊 Resultados del modelo", "ℹ️ Metodología"]
)

# ===========================================================================
# 1. INICIO (landing)
# ===========================================================================
with tab_inicio:
    st.subheader("El problema")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            """
            Los cronogramas se arman con duraciones estimadas *a ojo* y los atrasos se descubren
            **cuando ya ocurrieron**. Para entonces, la fecha comprometida está en riesgo y el margen
            de acción es mínimo.

            Un gerente de proyecto necesita responder tres preguntas **antes** de que sea tarde:
            1. ¿Qué tareas tienen más probabilidad de atrasarse?
            2. Si se atrasan, ¿cuánto impacto tendrán en la fecha final?
            3. ¿Dónde debo concentrar mi atención hoy?
            """
        )
    with c2:
        m = META["metricas_clasificacion"]
        a, b = st.columns(2)
        a.metric("Tareas reales analizadas", f"{META['n_tareas']:,}")
        b.metric("Organizaciones", "8")
        a.metric("ROC-AUC (tareas futuras)", f"{m['ROC-AUC']:.2f}")
        b.metric("Tareas atrasadas en los datos", f"{META['pct_atrasadas']:.0f} %")

    st.subheader("La solución")
    pasos = [
        ("1", "Carga tu cronograma", "Excel o CSV con tareas, responsables, fechas y dependencias. También puedes editarlo aquí mismo."),
        ("2", "El modelo evalúa cada tarea", "XGBoost estima la probabilidad de atraso y un Random Forest, los días; SHAP explica el porqué."),
        ("3", "Simula el proyecto", "2,000 escenarios Monte Carlo propagan los atrasos por las dependencias: probabilidad de cumplir, P50 y P80."),
        ("4", "Prioriza", "Ordena las tareas por riesgo × impacto en la fecha final: el cuello de botella primero."),
    ]
    cols = st.columns(4)
    for col, (n, titulo, texto) in zip(cols, pasos):
        col.markdown(
            f'<div class="tarjeta"><div class="paso">{n}</div><h4>{titulo}</h4>{texto}</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Lo que aprendió el modelo")
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(FIGURAS / "03_factores_riesgo.png"), caption="Tasa de atraso según factores de riesgo (EDA)")
    with c2:
        st.image(str(FIGURAS / "10_importancia.png"), caption="Factores más influyentes según SHAP")
    st.info(
        "**Hallazgo clave:** el riesgo se dispara con fechas sin holgura y con responsables con historial de "
        "atrasos. Las tareas **sin seguimiento** (sin comentarios ni ajustes) se atrasan más que las que se "
        "replanifican activamente."
    )

    st.subheader("Enlaces del proyecto")
    l1, l2, l3, l4 = st.columns(4)
    l1.link_button("💻 Repositorio GitHub", REPO_URL, width="stretch")
    l2.link_button("📄 Documento técnico (PDF)", DOC_URL, width="stretch")
    l3.link_button("📓 Notebook", NOTEBOOK_URL, width="stretch")
    l4.link_button("🎬 Video de presentación", VIDEO_URL, width="stretch")
    st.caption(f"Proyecto final · Machine Learning · Maestría en Inteligencia Artificial, Educación e Investigación · {AUTORA}")

# ===========================================================================
# 2. ANALIZAR CRONOGRAMA
# ===========================================================================
with tab_cron:
    st.subheader("Analiza tu cronograma")
    izq, der = st.columns([2, 1])
    with izq:
        origen = st.radio("Fuente de datos", ["Usar cronograma de ejemplo", "Subir mi archivo (Excel o CSV)"], horizontal=True)
    with der:
        st.download_button(
            "⬇️ Descargar plantilla", PLANTILLA.read_bytes(), "plantilla_cronograma.xlsx",
            help="Incluye una hoja de instrucciones con cada columna.", width="stretch",
        )

    crudo = None
    if origen.startswith("Usar"):
        crudo = pd.read_excel(EJEMPLO, sheet_name="cronograma")
        st.caption("Ejemplo: portal de autogestión para clientes · 20 tareas · 6 responsables. Puedes editar la tabla.")
        corte_def, comp_def = date(2026, 12, 1), date(2027, 3, 12)
    else:
        archivo = st.file_uploader("Cronograma", type=["xlsx", "xls", "csv"])
        st.caption("🔒 El archivo se procesa en memoria y no se guarda. Para datos sensibles, ejecuta la app en tu computadora.")
        if archivo is not None:
            try:
                crudo = pd.read_csv(archivo) if archivo.name.endswith(".csv") else pd.read_excel(archivo, sheet_name=0)
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo leer el archivo: {e}")
        corte_def, comp_def = date.today(), None

    if crudo is not None:
        for col in crudo.columns:
            if pd.api.types.is_datetime64_any_dtype(crudo[col]):
                crudo[col] = crudo[col].dt.date
        editado = st.data_editor(crudo, num_rows="dynamic", width="stretch", height=280, key=f"editor_{origen}")
        try:
            cron = preparar_cronograma(editado)
        except ErrorCronograma as e:
            st.error(f"⚠️ {e}")
            cron = None

        if cron is not None:
            with st.expander("⚙️ Parámetros del análisis", expanded=True):
                p1, p2, p3, p4 = st.columns(4)
                fecha_corte = p1.date_input("Fecha de corte (hoy)", corte_def, help="Fecha a la que corresponde el estado del cronograma.")
                fin_plan = cron["fecha_fin_planificada"].max().date()
                fecha_comp = p2.date_input("Fecha comprometida", comp_def or fin_plan, help="Fecha final acordada con el patrocinador.")
                inc = p3.selectbox("Incertidumbre de la simulación", list(INCERTIDUMBRE), index=1,
                                   help="Dispersión de los días de atraso en la simulación Monte Carlo.")
                escala = p4.slider("Ajuste a tu contexto", 0.25, 1.0, 1.0, 0.05,
                                   help="1.0 = magnitudes de atraso del dataset (proyectos open source). "
                                        "Si en tu organización los atrasos suelen ser más cortos, redúcelo.")

            fecha_corte, fecha_comp = pd.Timestamp(fecha_corte), pd.Timestamp(fecha_comp)
            var = calcular_variables(cron, fecha_corte, META["pct_atrasos_mediana"])
            pred = predecir(var, CLF, REG)
            sim = simular(pred, fecha_corte, INCERTIDUMBRE[inc], n=2000, escala=escala, fecha_compromiso=fecha_comp)
            prio = priorizar(pred, sim["criticidad"])
            pred["nivel"] = [nivel(p, c) for p, c in zip(pred["prob_atraso"], pred["cerrada"])]
            pred["criticidad"] = pred["id"].map(sim["criticidad"]).fillna(0)

            # ---- Indicadores
            st.markdown("### Pronóstico del proyecto")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Probabilidad de cumplir", f"{sim['prob_cumplir']:.0%}", help=f"P(terminar el {fecha_comp:%d/%m/%Y} o antes)")
            k2.metric("Fecha probable (P50)", f"{sim['p50']:%d/%m/%Y}", f"{(sim['p50'] - fecha_comp).days:+d} días vs. compromiso", delta_color="inverse")
            k3.metric("Fecha segura (P80)", f"{sim['p80']:%d/%m/%Y}", f"{(sim['p80'] - fecha_comp).days:+d} días", delta_color="inverse")
            abiertas = pred[~pred["cerrada"]]
            k4.metric("Tareas en riesgo alto", f"{(abiertas['nivel'] == 'Alto').sum()} de {len(abiertas)}")

            if sim["prob_cumplir"] >= 0.8:
                st.success("🟢 El proyecto tiene alta probabilidad de cumplir la fecha comprometida.")
            elif sim["prob_cumplir"] >= 0.5:
                st.warning("🟡 La fecha comprometida está en riesgo moderado. Revisa las tareas prioritarias.")
            else:
                st.error(f"🔴 Riesgo alto de incumplir la fecha. Una fecha con 80 % de confianza sería el **{sim['p80']:%d/%m/%Y}**.")

            # ---- Gantt
            st.markdown("### Cronograma: plan vs. estimación")
            g = pred.copy()
            g["ini_est"] = g["id"].map(sim["inicio_medio_tarea"])
            g["fin_est"] = g["id"].map(sim["fin_medio_tarea"])
            g = g.sort_values("fecha_inicio", ascending=False)
            g["etiqueta"] = [("★ " if c >= 0.5 else "") + f"{i}. {t}" for i, t, c in zip(g["id"], g["tarea"], g["criticidad"])]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                base=g["fecha_inicio"], x=(g["fecha_fin_planificada"] - g["fecha_inicio"]).dt.days * 86400000 + 86400000,
                y=g["etiqueta"], orientation="h", name="Plan", marker=dict(color="rgba(0,0,0,0)", line=dict(color=GRIS, width=1.5)),
                hovertemplate="Plan: %{base|%d/%m/%Y} → %{customdata|%d/%m/%Y}<extra></extra>", customdata=g["fecha_fin_planificada"],
            ))
            for nv in ["Alto", "Medio", "Bajo", "Cerrada"]:
                s = g[g["nivel"] == nv]
                if s.empty:
                    continue
                fig.add_trace(go.Bar(
                    base=s["ini_est"], x=(s["fin_est"] - s["ini_est"]).dt.days * 86400000 + 86400000, y=s["etiqueta"],
                    orientation="h", name=f"Estimado · riesgo {nv.lower()}" if nv != "Cerrada" else "Cerrada", width=0.5,
                    marker=dict(color=ESTADO[nv]), customdata=np.c_[s["fin_est"].dt.strftime("%d/%m/%Y"), (s["prob_atraso"] * 100).round(0), s["responsable"]],
                    hovertemplate="<b>%{y}</b><br>Fin estimado (P50): %{customdata[0]}<br>Riesgo: %{customdata[1]}%<br>Responsable: %{customdata[2]}<extra></extra>",
                ))
            fig.add_vline(x=fecha_comp, line=dict(color=NARANJA, width=2))
            fig.add_annotation(x=fecha_comp, y=1.02, yref="paper", text="Fecha comprometida", showarrow=False, font=dict(color="#0b0b0b"))
            fig.add_vline(x=fecha_corte, line=dict(color=GRIS, width=1, dash="dot"))
            fig.add_annotation(x=fecha_corte, y=-0.06, yref="paper", text="Hoy", showarrow=False, font=dict(color="#52514e"))
            fig.update_layout(barmode="overlay", height=max(380, 30 * len(g) + 120), xaxis=dict(type="date", gridcolor="#e1e0d9"),
                              plot_bgcolor="#fcfcfb", legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=40, b=40))
            st.plotly_chart(fig, width="stretch")
            st.caption("Contorno gris = plan · barra de color = estimación mediana del modelo · ★ = en la cadena crítica en ≥ 50 % de los escenarios.")

            # ---- Distribución Monte Carlo
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("### Escenarios simulados de fecha final")
                fechas = sim["origen"] + pd.to_timedelta(sim["fin_proyecto_dias"], unit="D")
                h = go.Figure(go.Histogram(x=fechas, nbinsx=40, marker_color=AZUL, hovertemplate="%{x}<br>%{y} escenarios<extra></extra>"))
                h.add_vline(x=fecha_comp, line=dict(color=NARANJA, width=2))
                h.add_annotation(x=fecha_comp, y=1.05, yref="paper", text="Comprometida", showarrow=False)
                h.add_vline(x=sim["p80"], line=dict(color="#0b0b0b", width=1, dash="dash"))
                h.add_annotation(x=sim["p80"], y=0.95, yref="paper", text="P80", showarrow=False, xanchor="left")
                h.update_layout(height=320, plot_bgcolor="#fcfcfb", margin=dict(l=10, r=10, t=30, b=30), yaxis_title="Escenarios",
                                xaxis=dict(gridcolor="#e1e0d9"), yaxis=dict(gridcolor="#e1e0d9"))
                st.plotly_chart(h, width="stretch")
            with c2:
                st.markdown("### Riesgo por responsable")
                r = (abiertas.groupby("responsable")
                     .agg(tareas=("id", "size"), riesgo_medio=("prob_atraso", "mean"), alto=("nivel", lambda s: (s == "Alto").sum()))
                     .sort_values("riesgo_medio", ascending=False))
                r["riesgo_medio"] = (r["riesgo_medio"] * 100).round(0).astype(int).astype(str) + " %"
                r.columns = ["Tareas abiertas", "Riesgo medio", "En riesgo alto"]
                st.dataframe(r, width="stretch")

            # ---- Prioridades
            st.markdown("### 🎯 Prioridades: dónde poner la atención")
            st.caption("Puntaje = atraso esperado (probabilidad × días) × peso en la ruta crítica. Las primeras son el cuello de botella.")
            tabla = prio.head(10).copy()
            tabla["Nivel"] = [f"{ICONO[nivel(p)]} {nivel(p)}" for p in tabla["prob_atraso"]]
            tabla = tabla.assign(
                Riesgo=(tabla["prob_atraso"] * 100).round(0).astype(int).astype(str) + " %",
                **{"Días si se atrasa": tabla["dias_atraso_si_ocurre"].round(0).astype(int),
                   "En ruta crítica": (tabla["criticidad"] * 100).round(0).astype(int).astype(str) + " %",
                   "Puntaje": tabla["puntaje_atencion"].round(1)},
            )[["id", "tarea", "responsable", "Nivel", "Riesgo", "Días si se atrasa", "En ruta crítica", "Puntaje"]]
            st.dataframe(tabla.rename(columns={"id": "ID", "tarea": "Tarea", "responsable": "Responsable"}),
                         width="stretch", hide_index=True)

            # ---- Explicación por tarea
            st.markdown("### ¿Por qué esta tarea está en riesgo?")
            opciones = prio["id"].tolist()
            if opciones:
                sel = st.selectbox("Elige una tarea", opciones,
                                   format_func=lambda i: f"{i}. {pred.set_index('id').loc[i, 'tarea']}")
                fila = pred[pred["id"] == sel]
                X = matriz_modelo(fila)
                cont = contribuciones(X)
                e1, e2 = st.columns([1, 2])
                p = float(fila["prob_atraso"].iloc[0])
                e1.metric("Probabilidad de atraso", f"{p:.0%}")
                e1.markdown(f"**Nivel:** {ICONO[nivel(p)]} {nivel(p)}")
                e1.metric("Días estimados si se atrasa", f"{fila['dias_atraso_si_ocurre'].iloc[0]:.0f}")
                e1.metric("Peso en la ruta crítica", f"{fila['criticidad'].iloc[0]:.0%}")
                e2.plotly_chart(grafico_explicacion(X.iloc[0], cont.iloc[0]), width="stretch")

            salida = pred[["id", "tarea", "responsable", "fecha_inicio", "fecha_fin_planificada", "estado", "nivel"]].copy()
            salida["prob_atraso_%"] = (pred["prob_atraso"] * 100).round(1)
            salida["dias_si_se_atrasa"] = pred["dias_atraso_si_ocurre"].round(1)
            salida["fin_estimado_P50"] = pred["id"].map(sim["fin_medio_tarea"]).dt.date
            salida["fin_estimado_P80"] = pred["id"].map(sim["fin_p80_tarea"]).dt.date
            salida["en_ruta_critica_%"] = (pred["criticidad"] * 100).round(0)
            st.download_button("⬇️ Descargar análisis en Excel", a_excel(salida), "analisis_cronograma.xlsx")

# ===========================================================================
# 3. EVALUAR UNA TAREA
# ===========================================================================
with tab_tarea:
    st.subheader("Evalúa una tarea antes de comprometer la fecha")
    st.caption("Útil en la planificación del sprint o al aceptar un nuevo compromiso.")
    f1, f2, f3 = st.columns(3)
    with f1:
        tipo = st.selectbox("Tipo de tarea", TIPOS, index=TIPOS.index("Nueva funcionalidad"))
        prioridad = st.select_slider("Prioridad", [1, 2, 3, 4, 5], 3,
                                     format_func=lambda v: {1: "Trivial", 2: "Baja", 3: "Media", 4: "Alta", 5: "Bloqueante"}[v])
        dias_lim = st.number_input("Días hasta la fecha límite", 0, 365, 10)
        dias_crea = st.number_input("Días desde que se creó la tarea", 0, 1000, 0)
    with f2:
        carga = st.number_input("Tareas abiertas del responsable", 0, 100, 5)
        pct = st.slider("% histórico de atrasos del responsable", 0, 100, int(META["pct_atrasos_mediana"]))
        preds_n = st.number_input("Tareas de las que depende", 0, 20, 0)
        bloquea = st.number_input("Tareas que dependen de esta", 0, 20, 0)
    with f3:
        reap = st.number_input("Reaperturas (retrabajo)", 0, 20, 0)
        coment = st.number_input("Comentarios / seguimientos", 0, 100, 1)
        replan = st.number_input("Replanificaciones", 0, 20, 1)
        alcance = st.number_input("Cambios de alcance", 0, 20, 0)

    fila = pd.DataFrame([{
        "tipo": tipo, "prioridad": prioridad, "dias_hasta_fecha_limite": dias_lim, "dias_desde_creacion": dias_crea,
        "carga_responsable": carga, "pct_atrasos_responsable": pct, "predecesoras": preds_n, "tareas_que_bloquea": bloquea,
        "tareas_relacionadas": preds_n + bloquea, "reaperturas": reap, "comentarios": coment, "cambios_prioridad": 0,
        "cambios_alcance": alcance, "entregas_asociadas": 1, "replanificaciones": replan, "dias_definicion": 0,
    }])
    X = matriz_modelo(fila)
    p = float(CLF.predict_proba(X)[0, 1])
    d = float(np.expm1(REG.predict(X)[0]))
    nv = nivel(p)
    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": " %"},
            title={"text": f"Riesgo de atraso: {ICONO[nv]} {nv}"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": ESTADO[nv]},
                   "steps": [{"range": [0, 30], "color": "#e7f6e7"}, {"range": [30, UMBRAL * 100], "color": "#fff4d6"},
                             {"range": [UMBRAL * 100, 100], "color": "#fbe4e4"}]},
        ))
        gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, width="stretch")
        st.metric("Días estimados de atraso si ocurre", f"{d:.0f}")
        st.metric("Atraso esperado", f"{p * d:.1f} días")
    with r2:
        st.markdown("**¿Qué explica este resultado?**")
        st.plotly_chart(grafico_explicacion(X.iloc[0], contribuciones(X).iloc[0]), width="stretch")
    if nv == "Alto":
        sugerencias = []
        if dias_lim <= 3:
            sugerencias.append("Dar más holgura a la fecha límite.")
        if pct > 25:
            sugerencias.append("Revisar con el responsable su carga real y su forma de estimar.")
        if reap > 0:
            sugerencias.append("Atacar la causa del retrabajo antes de continuar.")
        if coment == 0 and replan == 0:
            sugerencias.append("Darle seguimiento activo: las tareas sin actividad tienden a atrasarse.")
        if preds_n > 0:
            sugerencias.append("Asegurar que las dependencias se cierren a tiempo.")
        if sugerencias:
            st.warning("**Acciones sugeridas:**\n\n" + "\n".join(f"- {s}" for s in sugerencias))

# ===========================================================================
# 4. RESULTADOS DEL MODELO
# ===========================================================================
with tab_result:
    st.subheader("Desempeño de los modelos")
    st.markdown(
        f"Validación **temporal**: entrenamiento con {META['n_entrenamiento']:,} tareas (2004–2014) y prueba con "
        f"{META['n_prueba']:,} tareas posteriores (2014–2016) que el modelo nunca vio."
    )
    st.markdown("**Modelo 1 · Clasificación: ¿la tarea se atrasará?**")
    tc = pd.DataFrame(META["test_clasificacion"]).set_index("modelo")
    st.dataframe(tc.style.format("{:.3f}").highlight_max(subset=["ROC-AUC", "PR-AUC", "F1"], color="#cde2fb"), width="stretch")
    st.markdown("**Modelo 2 · Regresión: si se atrasa, ¿cuántos días?** (MAE y MedAE en días)")
    tr = pd.DataFrame(META["test_regresion"]).set_index("modelo")
    st.dataframe(tr.style.format("{:.2f}"), width="stretch")
    st.caption(f"Modelos seleccionados por validación cruzada temporal: {META['modelo_clasificacion']} (clasificación) y "
               f"{META['modelo_regresion']} (regresión). Umbral de decisión: {UMBRAL:.2f}.")
    figs = [
        ("07_evaluacion_clasificacion.png", "Curvas ROC, Precisión-Recall y matriz de confusión"),
        ("09_shap_resumen.png", "Interpretación SHAP"),
        ("08_calibracion.png", "Calibración de probabilidades"),
        ("11_regresion.png", "Modelo de regresión: real vs. estimado"),
        ("01_balance_y_proyectos.png", "Balance de clases y tasa por organización"),
        ("02_distribucion_atraso.png", "Distribución de los días de atraso"),
    ]
    st.image(str(FIGURAS / figs[0][0]), caption=figs[0][1])
    for a, b in [(1, 2), (3, 4)]:
        c1, c2 = st.columns(2)
        c1.image(str(FIGURAS / figs[a][0]), caption=figs[a][1])
        c2.image(str(FIGURAS / figs[b][0]), caption=figs[b][1])
    st.image(str(FIGURAS / figs[5][0]), caption=figs[5][1])

# ===========================================================================
# 5. METODOLOGÍA
# ===========================================================================
with tab_acerca:
    st.subheader("Metodología")
    st.markdown(
        """
        **Datos.** 64,747 tareas reales con fecha límite de los sistemas JIRA de 8 organizaciones de software
        (Apache, Duraspace, Java.net, JBoss, JIRA, Moodle, Mulesoft, WSO2), publicadas por Choetkiertikul et al. (2017).
        Las variables se midieron **al momento de asignar la fecha límite**, sin información del futuro.

        **Proceso (CRISP-DM).** Comprensión del problema → limpieza y traducción de variables → análisis exploratorio →
        validación cruzada temporal de 3 clasificadores y 3 regresores → selección → interpretación SHAP →
        integración en la app con simulación Monte Carlo.

        **Simulación.** En cada uno de 2,000 escenarios, cada tarea abierta se atrasa con la probabilidad del modelo y, si
        se atrasa, lo hace alrededor de los días estimados. Los atrasos se propagan a las tareas dependientes. Así se obtienen
        la probabilidad de cumplir, las fechas P50/P80 y el índice de criticidad de cada tarea.
        """
    )
    st.markdown("**Variables que usa el modelo**")
    st.dataframe(pd.DataFrame({"Variable": list(DESCRIPCION), "Descripción": list(DESCRIPCION.values())}),
                 hide_index=True, width="stretch")
    st.subheader("Limitaciones")
    st.markdown(
        """
        - El modelo aprendió de proyectos **de software de código abierto**; en otros contextos (banca, educación,
          construcción) las predicciones son **orientativas** y conviene recalibrarlas con historial propio.
        - Estimar **cuántos días** se atrasará una tarea es mucho más incierto que estimar **si** se atrasará;
          por eso se comunican rangos (P50/P80) y no una fecha única.
        - Algunas relaciones son de **correlación, no de causalidad** (por ejemplo, más replanificaciones ↔ menos atraso,
          porque indican seguimiento activo).
        - Los datos que el usuario no registra (comentarios, replanificaciones…) se completan con valores típicos.
        """
    )
    st.subheader("Referencias principales")
    st.markdown(
        """
        - Choetkiertikul, M., Dam, H. K., Tran, T., & Ghose, A. (2017). Predicting the delay of issues with due dates in
          software projects. *Empirical Software Engineering, 22*(3), 1223–1263. https://doi.org/10.1007/s10664-016-9496-7
        - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD '16*, 785–794.
        - Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
        - Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
        """
    )
