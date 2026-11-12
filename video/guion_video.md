# Guion del video de presentación (≈ 5 minutos)

**Proyecto:** Asistente inteligente de cronogramas: predicción de retrasos en tareas de proyectos mediante Machine Learning
**Presenta:** Sheila De Oleo

> Sugerencia: graba la pantalla con la app abierta (pestañas en el orden indicado) y tu cámara en una esquina. Herramientas posibles: OBS, Loom, Clipchamp o Canva.

---

### 1. Apertura y problema (0:00 – 0:40)
**En pantalla:** pestaña *Inicio* de la app.

> "¿Cuántas veces nos enteramos de que una tarea se atrasó cuando ya era tarde? En gestión de proyectos, los cronogramas se arman con estimaciones a ojo y los atrasos se descubren cuando ya ocurrieron. Mi proyecto busca cambiar eso: usar Machine Learning para anticipar qué tareas se van a atrasar, cuánto, y dónde debe concentrarse el gerente para cumplir la fecha comprometida."

### 2. Datos (0:40 – 1:20)
**En pantalla:** notebook, sección 1 (tabla por organización) y gráfico de balance.

> "Trabajé con datos reales y públicos: casi 65 mil tareas con fecha límite de ocho organizaciones de software, publicados en la revista *Empirical Software Engineering*. Un detalle clave es que las variables se midieron en el momento en que se asignó la fecha límite, igual que cuando un gerente compromete una fecha. Así el modelo no usa información del futuro. El 21 % de las tareas se atrasó, así que es un problema desbalanceado."

### 3. Hallazgos del análisis exploratorio (1:20 – 2:00)
**En pantalla:** figura de factores de riesgo.

> "El análisis exploratorio mostró que las tareas comprometidas con 0 o 1 día de holgura se atrasan hasta en un 77 % de los casos, y que el historial del responsable pesa mucho. Pero hubo un hallazgo contraintuitivo: las tareas que nunca se replanifican se atrasan más que las replanificadas. ¿Por qué? Porque replanificar es señal de que alguien está dando seguimiento. Las tareas olvidadas son las que se atrasan. Correlación no es causalidad."

### 4. Modelos y resultados (2:00 – 3:00)
**En pantalla:** tabla de métricas, curvas ROC/PR y SHAP.

> "Comparé regresión logística, Random Forest y XGBoost con validación cruzada temporal: siempre entrenando con el pasado y validando con el futuro. XGBoost fue el mejor, con un ROC-AUC de 0.95 sobre tareas que nunca vio. Como las clases están desbalanceadas, también miré el PR-AUC, de 0.86, frente a 0.20 de un modelo al azar. Ajusté el umbral en lugar de ponderar clases, para que las probabilidades quedaran calibradas.
> Para estimar cuántos días se atrasará una tarea usé un segundo modelo. Fue honestamente más difícil: mejora frente a la línea base, pero con mucha incertidumbre. Por eso la app comunica rangos y no una fecha exacta.
> Con SHAP el modelo deja de ser una caja negra: el historial del responsable, la holgura y el nivel de seguimiento son los factores que más pesan."

### 5. Demostración del prototipo (3:00 – 4:30)
**En pantalla:** pestaña *Analizar cronograma* con el ejemplo.

> "Este es el prototipo. Cargo un cronograma: un portal de autogestión para clientes con 20 tareas y sus dependencias. El plan termina el 12 de febrero y comprometimos el 12 de marzo."

*(Señalar los indicadores)*
> "La app simula dos mil escenarios propagando los atrasos por las dependencias. Resultado: solo 22 % de probabilidad de cumplir, y una fecha con 80 % de confianza a finales de abril."

*(Bajar al Gantt)*
> "El Gantt compara el plan con la estimación. En rojo, las tareas de riesgo alto, y con estrella, las que están en la ruta crítica."

*(Bajar a Prioridades y a la explicación)*
> "Lo más útil: la lista de prioridades. La integración con el core y el módulo de solicitudes son el cuello de botella. Y aquí veo por qué está en riesgo cada tarea."

*(Pestaña Evaluar una tarea, mover la holgura a 1 día)*
> "También puedo evaluar una tarea antes de comprometer la fecha: si le doy solo un día de holgura, el riesgo se dispara."

### 6. Cierre (4:30 – 5:00)
**En pantalla:** README del repositorio.

> "En resumen: pasé de los datos a un modelo, del modelo a un prototipo y del prototipo a una herramienta que puedo usar en mi día a día como gerente de proyectos. Como trabajo futuro, la idea es reentrenarlo con el historial de mi organización y conectarlo con Jira o Azure DevOps. Todo el código, el notebook y el documento están en el repositorio. ¡Gracias!"
