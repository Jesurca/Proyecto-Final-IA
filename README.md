# 🚢 Titanic ML Dashboard

Dashboard interactivo de Machine Learning sobre el dataset del Titanic.  
Desarrollado con **Streamlit** + **scikit-learn**.

## 📦 Instalación

```bash
pip install -r requirements.txt
```

## 🚀 Ejecución

```bash
streamlit run app.py
```

Abre automáticamente en: `http://localhost:8501`

## 📁 Estructura del proyecto

```
titanic_app/
├── app.py           # Dashboard principal (Streamlit)
├── model.py         # Lógica de ML: limpieza, clasificación, PCA, clustering
├── requirements.txt # Dependencias Python
└── README.md        # Este archivo
```

## 🧠 Secciones del Dashboard

| Sección | Contenido |
|---|---|
| 🏠 Inicio & EDA | Exploración y estadísticas del dataset |
| 🤖 Clasificación | Logistic Regression vs Random Forest |
| 📐 PCA | Reducción de dimensionalidad |
| 🔵 Clustering | K-Means con K=2,3,4 sobre 3 conjuntos |
| 🏆 Mejor Modelo | Selección final y conclusiones |
| 🔮 Predictor | Predicción individual de supervivencia |

## 📊 Resultados clave

- **Mejor clasificador:** Random Forest (Acc: 83.2%, AUC: 0.894)
- **Mejor clustering:** X3 (Pclass, Sex, Age), K-Means, K=4 (Silhouette: 0.57)
- **Variables más importantes:** Sex > FareLog > Age
