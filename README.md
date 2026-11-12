# 📅 Asistente inteligente de cronogramas

**Predicción de retrasos en tareas de proyectos mediante Machine Learning**

Proyecto final · Curso de Machine Learning · Maestría en Inteligencia Artificial, Educación e Investigación
**Autora:** Sheila De Oleo

| 🌐 Aplicación web | 📄 Documento técnico | 📓 Notebook | 🎬 Video |
|---|---|---|---|
| [Abrir la app](https://ENLACE-DE-LA-APP.streamlit.app) | [PDF](docs/Documento_Tecnico.pdf) | [Cuaderno de trabajo](notebooks/01_prediccion_retrasos_cronogramas.ipynb) | [Ver video](https://youtu.be/ENLACE_DEL_VIDEO) |

---

## 🎯 Problemática

Los cronogramas se arman con duraciones estimadas "a ojo" y los atrasos se descubren **cuando ya ocurrieron**. Este proyecto responde, **antes** de que sea tarde, tres preguntas clave para un gerente de proyecto:

1. ¿Qué tareas tienen más probabilidad de atrasarse?
2. Si se atrasan, ¿cuánto impacto tendrán en la fecha final?
3. ¿Dónde debo concentrar mi atención hoy?

## 💡 Solución

| Componente | Técnica | Resultado (tareas futuras, nunca vistas) |
|---|---|---|
| **¿Se atrasará la tarea?** | Clasificación con XGBoost (comparado con regresión logística y Random Forest) | ROC-AUC **0.95** · PR-AUC **0.86** · F1 **0.76** |
| **¿Cuántos días?** | Regresión con Random Forest (comparado con Ridge y XGBoost) | Error mediano **14.6 días** (línea base: 22) |
| **¿Por qué?** | Interpretación con SHAP | Factores de riesgo accionables |
| **¿Cumpliremos la fecha?** | Simulación Monte Carlo sobre las dependencias del cronograma | Probabilidad de cumplir, fechas P50/P80, ruta crítica |

**Hallazgos principales:** el riesgo se dispara con fechas sin holgura (0–1 días: hasta 77 % de atraso) y con responsables con historial de atrasos. Las tareas **sin seguimiento** (sin comentarios ni replanificaciones) se atrasan más que las que se gestionan activamente.

## 📊 Datos

**64,747 tareas reales con fecha límite** de los sistemas JIRA de 8 organizaciones de software (Apache, Duraspace, Java.net, JBoss, JIRA, Moodle, Mulesoft y WSO2), publicadas por:

> Choetkiertikul, M., Dam, H. K., Tran, T., & Ghose, A. (2017). Predicting the delay of issues with due dates in software projects. *Empirical Software Engineering, 22*(3), 1223–1263. https://doi.org/10.1007/s10664-016-9496-7

Las variables se midieron **en el momento en que se asignó la fecha límite**, sin información del futuro. Validación **temporal**: entrenamiento con tareas de 2004–2014 y prueba con tareas de 2014–2016.

## 🖥️ La aplicación

- **Inicio:** presentación del problema, la solución y los hallazgos.
- **Analizar cronograma:** sube tu Excel/CSV (o usa el ejemplo), y obtén el pronóstico del proyecto, un Gantt de plan vs. estimación, el riesgo por responsable, las tareas prioritarias y la explicación de cada riesgo.
- **Evaluar una tarea:** estima el riesgo de una tarea antes de comprometer su fecha.
- **Resultados del modelo:** métricas y gráficos de evaluación.
- **Metodología:** datos, proceso, limitaciones y referencias.

### Usarla con tu propio cronograma

Descarga la plantilla desde la app (`data/plantilla_cronograma.xlsx`). Columnas obligatorias:

| Columna | Ejemplo |
|---|---|
| `id` | 7 |
| `tarea` | Integración con core (API) |
| `tipo` | Bug, Mejora, Tarea, Subtarea, Nueva funcionalidad, Prueba, Historia de usuario, Documentación |
| `prioridad` | Trivial, Baja, Media, Alta, Bloqueante |
| `responsable` | Carlos Méndez |
| `fecha_inicio` / `fecha_fin_planificada` | 2026-12-02 / 2026-12-23 |

Opcionales: `predecesoras` (ej. `3,4`), `estado` (Pendiente / En progreso / Completada), `fecha_fin_real` (permite calcular el historial de cada responsable), `reaperturas`, `comentarios`, `cambios_alcance`, `replanificaciones`.

🔒 Los archivos se procesan en memoria y no se guardan. Para cronogramas con información sensible, ejecuta la app localmente.

## 🚀 Instrucciones de ejecución

```bash
git clone https://github.com/sheiladeoleo12-star/Proyecto-UFHEC-ml.git
cd Proyecto-UFHEC-ml
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate

# Aplicación web
pip install -r requirements.txt
streamlit run app.py               # se abre en http://localhost:8501

# Notebook (reentrena los modelos y regenera las figuras)
pip install -r requirements-notebook.txt
jupyter notebook notebooks/01_prediccion_retrasos_cronogramas.ipynb
```

Probado con Python 3.12–3.13.

## 📁 Estructura del repositorio

```
├── app.py                         # Aplicación web (Streamlit)
├── src/
│   ├── features.py                # Limpieza y variables (compartido por notebook y app)
│   └── cronograma.py              # Lectura del cronograma, predicción, Monte Carlo y priorización
├── notebooks/
│   └── 01_prediccion_retrasos_cronogramas.ipynb   # EDA, modelado, evaluación, SHAP
├── models/                        # Modelos entrenados + metadata.json (métricas, umbral)
├── data/
│   ├── raw/                       # Dataset original (8 organizaciones) + FUENTE.md
│   ├── ejemplo_cronograma.xlsx    # Cronograma de ejemplo para la demo
│   └── plantilla_cronograma.xlsx  # Plantilla para usuarios
├── reports/
│   ├── figures/                   # Gráficos generados por el notebook
│   └── *.csv                      # Tablas de métricas
├── docs/
│   └── Documento_Tecnico.pdf      # Documento técnico (normas APA)
├── video/
│   └── guion_video.md             # Guion del video de presentación
├── requirements.txt               # Dependencias de la app
└── requirements-notebook.txt      # Dependencias del notebook
```

## 🛠️ Tecnologías

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · Plotly · matplotlib / seaborn · Jupyter

## ⚠️ Limitaciones

- El modelo aprendió de proyectos de software de código abierto. En otros contextos (banca, educación) las predicciones son **orientativas** y conviene recalibrarlas con el historial propio. La app incluye un *factor de ajuste al contexto*.
- Estimar **cuántos** días se atrasará una tarea es mucho más incierto que estimar **si** se atrasará. Por eso la app comunica rangos (P50/P80).
- Algunas relaciones son de correlación, no de causalidad.

## 📚 Referencias principales

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD '16*, 785–794.
- Choetkiertikul, M., Dam, H. K., Tran, T., & Ghose, A. (2017). Predicting the delay of issues with due dates in software projects. *Empirical Software Engineering, 22*(3), 1223–1263.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.

La lista completa en formato APA está en el [documento técnico](docs/Documento_Tecnico.pdf).
