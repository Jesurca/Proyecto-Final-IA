"""
app.py — Dashboard Titanic: Clasificación, PCA y Clustering
Ejecutar: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Importar lógica de ML
from model import (
    load_and_clean,
    train_classifiers,
    run_pca,
    run_clustering,
    get_best_clustering,
    predict_passenger,
    FEATURE_LABELS,
    CLUSTER_DESCRIPTIONS,
    FEATURE_SETS,
)

# ══════════════════════════════════════════════
# CONFIGURACIÓN PÁGINA
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Titanic · JDUC · IA 2026",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {font-size: 1rem; color: #666; margin-bottom: 1.5rem;}
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 12px; padding: 1rem 1.5rem;
        color: white; text-align: center; margin: 0.3rem;
    }
    .winner-badge {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white; border-radius: 8px; padding: 0.4rem 0.8rem;
        font-weight: bold; display: inline-block;
    }
    .cluster-card {
        background: #f8f9fa; border-left: 4px solid #667eea;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0;
    }
    div[data-testid="stMetric"] {
        background: #1e1e26; 
        border-radius: 10px;
        padding: 0.5rem 1rem; 
        border: 1px solid #3a3b46;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# CACHÉ — cargar datos y modelos una sola vez
# ══════════════════════════════════════════════

@st.cache_data(show_spinner="⏳ Descargando y limpiando datos del Titanic…")
def load_data():
    return load_and_clean()

@st.cache_resource(show_spinner="🤖 Entrenando modelos de clasificación…")
def get_models(df):
    return train_classifiers(df)

@st.cache_data(show_spinner="📐 Ejecutando PCA…")
def get_pca(df):
    return run_pca(df)

@st.cache_data(show_spinner="🔵 Ejecutando clustering…")
def get_clustering(df):
    return run_clustering(df), get_best_clustering(df)

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/St%C3%B6wer_Titanic.jpg/330px-St%C3%B6wer_Titanic.jpg",
             use_container_width=True, caption="RMS Titanic, 1912")
    st.markdown("## 🧭 Navegación")
    page = st.radio(
        "Sección",
        ["🏠 Inicio & EDA",
         "🤖 Modelos de Clasificación",
         "📐 Reducción PCA",
         "🔵 Clustering K-Means",
         "🏆 Mejor Modelo & Conclusiones",
         "🔮 Predictor Individual"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset:** Titanic (Kaggle)  \n**Registros:** 891  \n**Variables:** 12")
    st.markdown("**Modelos:** Logistic Reg. · Random Forest")
    st.markdown("**Clustering:** K-Means · K=2,3,4")
    st.markdown("Por:")
    st.markdown("Jesús David Urbiñez Castro")
    st.markdown("Santiago Brito")
    
# ══════════════════════════════════════════════
# CARGAR TODO
# ══════════════════════════════════════════════
df = load_data()
lr, rf, scaler, clf_metrics, X_test, X_test_sc, y_test = get_models(df)
pca_data = get_pca(df)
cluster_table, best_clust = get_clustering(df)

# Añadir cluster al df para análisis
df_c = df.copy()
df_c["Cluster"] = best_clust["labels"]

# ══════════════════════════════════════════════
# PÁGINA 1 — INICIO & EDA
# ══════════════════════════════════════════════
if page == "🏠 Inicio & EDA":
    st.markdown('<div class="main-header">🚢 Kaggle: Titanic Dataset · Por: Jesús Urbiñez y Santigo Brito</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Análisis completo: EDA · Clasificación · PCA · Clustering — Proyecto IA 2026</div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pasajeros", f"{len(df):,}")
    c2.metric("Sobrevivieron", f"{df['Survived'].sum():,}", f"{df['Survived'].mean()*100:.1f}%")
    c3.metric("Edad promedio", f"{df['Age'].mean():.1f} años")
    c4.metric("Fare promedio", f"${df['Fare'].mean():.1f}")
    c5.metric("Variables usadas", "8")

    st.markdown("---")

    # Dataset preview
    with st.expander("📋 Vista del dataset limpio", expanded=False):
        st.dataframe(df[["Survived","Pclass","Sex","Age","Fare","Embarked","FamilySize","FareLog"]].head(20),
                     use_container_width=True)

    # Descripción
    with st.expander("📊 Estadísticas descriptivas"):
        st.dataframe(df[["Age","Fare","FamilySize","FareLog"]].describe().round(2), use_container_width=True)

    st.subheader("📈 Análisis Exploratorio de Datos")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,3.5))
        vals = df['Survived'].value_counts()
        ax.bar(['No Sobrevivió','Sobrevivió'], vals.values,
               color=['#e74c3c','#2ecc71'], edgecolor='black', linewidth=0.8)
        ax.set_title('Distribución de Supervivencia', fontweight='bold')
        ax.set_ylabel('Cantidad de Pasajeros')
        for i,v in enumerate(vals.values):
            ax.text(i, v+5, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5,3.5))
        surv_sex = df.groupby('Sex')['Survived'].mean()
        ax.bar(surv_sex.index, surv_sex.values,
               color=['#e91e63','#3498db'], edgecolor='black', linewidth=0.8)
        ax.set_title('Tasa de Supervivencia por Sexo', fontweight='bold')
        ax.set_ylabel('Tasa de Supervivencia')
        ax.set_ylim(0,1)
        for i,(lbl,v) in enumerate(surv_sex.items()):
            ax.text(i, v+0.02, f'{v:.2%}', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(5,3.5))
        surv_pclass = df.groupby('Pclass')['Survived'].mean()
        ax.bar(['1ª Clase','2ª Clase','3ª Clase'], surv_pclass.values,
               color=['#f1c40f','#9b59b6','#e74c3c'], edgecolor='black', linewidth=0.8)
        ax.set_title('Tasa de Supervivencia por Clase', fontweight='bold')
        ax.set_ylabel('Tasa de Supervivencia')
        ax.set_ylim(0,1)
        for i,v in enumerate(surv_pclass.values):
            ax.text(i, v+0.02, f'{v:.2%}', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(5,3.5))
        for survived, label, color in [(0,'No Sobrevivió','#e74c3c'),(1,'Sobrevivió','#2ecc71')]:
            ax.hist(df[df['Survived']==survived]['Age'], bins=20, alpha=0.6,
                    label=label, color=color, edgecolor='white')
        ax.set_title('Distribución de Edad por Supervivencia', fontweight='bold')
        ax.set_xlabel('Edad')
        ax.set_ylabel('Conteo')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Matriz de Confusión - Evaluación del Modelo
    st.subheader("📊 Matriz de Confusión")
    fig, ax = plt.subplots(figsize=(6, 5)) # Un tamaño más cuadrado ideal para matrices 2x2
    
    # 1. Calcular la matriz de confusión usando el modelo Random Forest entrenado
    # y_test: Valores reales del conjunto de prueba
    # y_pred: Predicciones del modelo sobre X_test
    from sklearn.metrics import confusion_matrix
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Etiquetas para los ejes
    labels = ["No Sobrevive (0)", "Sobrevive (1)"]
    
    # 2. Dibujar el heatmap de la matriz
    # Quitamos la máscara (mask) porque en la matriz necesitas ver los 4 cuadrantes
    sns.heatmap(cm, ax=ax, cmap='rocket', # 'rocket' te dará los tonos oscuros/rosados de tu imagen original
                annot=True, fmt='d', square=True, linewidths=0.5,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 12, "weight": "bold"})
    
    # Ajustes estéticos de títulos y etiquetas
    ax.set_title("Random Forest Confusion Matrix", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Predicción (Predicted Model)", fontsize=11, labelpad=10)
    ax.set_ylabel("Realidad (Actual Ground Truth)", fontsize=11, labelpad=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 3. Calcular métricas para el cuadro informativo (Opcional, pero le da un toque muy pro)
    tn, fp, fn, tp = cm.ravel()
    st.info(f"💡 **Interpretación de la Matriz:**\n"
            f"* **Verdaderos Negativos (TN):** {tn} pasajeros que no sobrevivieron fueron clasificados correctamente.\n"
            f"* **Verdaderos Positivos (TP):** {tp} pasajeros que sobrevivieron fueron identificados correctamente.\n"
            f"* **Falsos Positivos (FP):** {fp} se predijeron como sobrevivientes pero fallecieron.\n"
            f"* **Falsos Negativos (FN):** {fn} se predijeron como fallecidos pero sobrevivieron.")

# ══════════════════════════════════════════════
# PÁGINA 2 — MODELOS DE CLASIFICACIÓN
# ══════════════════════════════════════════════
elif page == "🤖 Modelos de Clasificación":
    st.title("🤖 Modelos de Clasificación")
    st.markdown("Comparativa entre **Regresión Logística** y **Random Forest** para predecir supervivencia.")

    # Métricas resumen
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown("### Regresión Logística")
        st.metric("Accuracy", f"{clf_metrics['lr']['accuracy']:.4f}")
        st.metric("AUC-ROC", f"{clf_metrics['lr']['auc']:.4f}")
        st.metric("CV Accuracy (5-fold)", f"{clf_metrics['lr']['cv_accuracy']:.4f}")
    with col2:
        st.markdown("### Random Forest")
        st.metric("Accuracy", f"{clf_metrics['rf']['accuracy']:.4f}")
        st.metric("AUC-ROC", f"{clf_metrics['rf']['auc']:.4f}")
        st.metric("CV Accuracy (5-fold)", f"{clf_metrics['rf']['cv_accuracy']:.4f}")
    with col3:
        st.markdown("### 🏆 Ganador")
        winner = clf_metrics['winner']
        winner_name = "Random Forest" if winner=="rf" else "Random Forest"
        st.markdown(f'<div class="winner-badge">🥇 {winner_name}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        **Random Forest supera a Logistic Regression en:**
        - Mayor AUC-ROC ({clf_metrics['rf']['auc']:.4f} vs {clf_metrics['lr']['auc']:.4f})
        - Mayor Accuracy ({clf_metrics['rf']['accuracy']:.4f} vs {clf_metrics['lr']['accuracy']:.4f})
        - Mayor CV Accuracy ({clf_metrics['rf']['cv_accuracy']:.4f} vs {clf_metrics['lr']['cv_accuracy']:.4f})
        """)

    st.markdown("---")

    # Confusion matrices
    st.subheader("📊 Matrices de Confusión")
    col1, col2 = st.columns(2)
    for col, (name, cm, acc) in zip([col1, col2], [
        ("Regresión Logística", clf_metrics['lr']['cm'], clf_metrics['lr']['accuracy']),
        ("Random Forest", clf_metrics['rf']['cm'], clf_metrics['rf']['accuracy']),
    ]):
        with col:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No (pred)', 'Sí (pred)'],
                        yticklabels=['No (real)', 'Sí (real)'],
                        linewidths=1, linecolor='white')
            ax.set_title(f'{name}\nAccuracy: {acc:.4f}', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Feature Importance
    st.subheader("📌 Importancia de Características — Random Forest")
    fi = clf_metrics['rf']['feature_importances']
    fig, ax = plt.subplots(figsize=(9,4))
    colors = ['#e74c3c' if v == fi.max() else '#3498db' for v in fi.values]
    fi.sort_values().plot(kind='barh', ax=ax, color=colors[::-1], edgecolor='white')
    ax.set_title('Importancia de Características (Random Forest)', fontweight='bold')
    ax.set_xlabel('Importancia Relativa')
    for i, v in enumerate(fi.sort_values().values):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("🔍 **Sex** (género) es la variable más importante, seguida de **FareLog** (tarifa) y **Age** (edad). "
            "Esto coincide con la regla histórica: mujeres y niños primero.")

    # Reporte clasificación
    with st.expander("📋 Reporte de Clasificación completo — Random Forest"):
        report = clf_metrics['rf']['report']
        rep_df = pd.DataFrame(report).T.round(3)
        st.dataframe(rep_df, use_container_width=True)


# ══════════════════════════════════════════════
# PÁGINA 3 — PCA
# ══════════════════════════════════════════════
elif page == "📐 Reducción PCA":
    st.title("📐 Reducción de Dimensionalidad — PCA")

    ev = pca_data["explained_ratio"]
    cum = pca_data["cum_var"]
    n85 = pca_data["n_components_85"]

    st.markdown(f"""
    **Objetivo:** Reducir las 8 variables originales a un espacio de menor dimensión preservando la varianza.

    Con **{n85} componentes principales** se explica el **≥85%** de la varianza total del dataset.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Componentes para 85% varianza", n85)
    col2.metric("Varianza PC1", f"{ev[0]*100:.1f}%")
    col3.metric("Varianza PC2", f"{ev[1]*100:.1f}%")

    st.markdown("---")
    st.subheader("📊 Varianza Explicada por Componente")
    fig, axes = plt.subplots(1,2, figsize=(13,5))

    ax = axes[0]
    ax.bar(range(1,9), ev*100, color='#667eea', alpha=0.8, edgecolor='white', label='Individual')
    ax.plot(range(1,9), cum*100, 'r-o', linewidth=2, label='Acumulada')
    ax.axhline(85, linestyle='--', color='green', linewidth=1.5, label='Umbral 85%')
    ax.axvline(n85, linestyle=':', color='orange', linewidth=1.5, label=f'PC{n85} = 85%')
    ax.set_xlabel('Componente Principal')
    ax.set_ylabel('Varianza Explicada (%)')
    ax.set_title('Scree Plot — Varianza por Componente')
    ax.set_xticks(range(1,9))
    ax.legend()
    ax.set_ylim(0,105)

    # PCA 2D coloreado por supervivencia
    X_pca = pca_data["X_pca"]
    y_vals = df["Survived"].values
    ax = axes[1]
    for survived, label, color, marker in [(0,'No Sobrevivió','#e74c3c','o'),(1,'Sobrevivió','#2ecc71','^')]:
        mask = y_vals == survived
        ax.scatter(X_pca[mask,0], X_pca[mask,1], c=color, label=label,
                   alpha=0.5, s=35, edgecolors='k', linewidths=0.3, marker=marker)
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% var.)')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% var.)')
    ax.set_title('Proyección PCA 2D — Color: Supervivencia Real')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Loadings table
    with st.expander("📋 Loadings de las dos primeras componentes"):
        pca2 = pca_data["pca2"]
        loadings = pd.DataFrame(
            pca2.components_.T,
            index=FEATURE_LABELS,
            columns=["PC1","PC2"]
        ).round(3)
        st.dataframe(loadings.style.background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)
        st.caption("Los loadings muestran la contribución de cada variable a cada componente principal.")

    st.info(f"💡 **PC1** ({ev[0]*100:.1f}%) captura principalmente variación en tarifa/clase. "
            f"**PC2** ({ev[1]*100:.1f}%) captura variación en edad y tamaño familiar.")


# ══════════════════════════════════════════════
# PÁGINA 4 — CLUSTERING
# ══════════════════════════════════════════════
elif page == "🔵 Clustering K-Means":
    st.title("🔵 Clustering — K-Means")
    st.markdown("Aplicación de K-Means con **K=2, 3, 4** sobre tres conjuntos de características, evaluando con Silhouette y Davies-Bouldin.")

    # Tabla de resultados
    st.subheader("📊 Comparativa de Configuraciones")
    styled = cluster_table.style.highlight_max(subset=["Silhouette ↑"], color="#c8f7c5")\
                                 .highlight_min(subset=["Davies-Bouldin ↓"], color="#c8f7c5")
    st.dataframe(styled, use_container_width=True)

    st.success("🏆 **Mejor configuración:** X3 (Pclass, Sex, Age) — K-Means — K=4  \n"
               f"Silhouette: {best_clust['silhouette']:.4f} | Davies-Bouldin: {best_clust['db']:.4f}")

    st.markdown("---")
    st.subheader("🎯 Visualización Final — K-Means K=4 sobre X3")

    X_pca_c = best_clust["X_pca"]
    labels = best_clust["labels"]
    y_real = df["Survived"].values

    COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12']

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,5))
        for k in range(4):
            mask = labels == k
            ax.scatter(X_pca_c[mask,0], X_pca_c[mask,1],
                       c=COLORS[k], label=f'Clúster {k}',
                       alpha=0.7, s=50, edgecolors='k', linewidths=0.3)
        centers = best_clust["km"].cluster_centers_
        ax.scatter(centers[:,0], centers[:,1], c='black', s=150,
                   marker='X', zorder=5, label='Centroides')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('K-Means K=4 — Espacio PCA (PC1 vs PC2)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6,5))
        surv_by_cluster = df_c.groupby("Cluster")["Survived"].mean()
        bars = ax.bar(range(4), surv_by_cluster.values, color=COLORS, edgecolor='black', linewidth=0.8)
        ax.set_xlabel('Clúster')
        ax.set_ylabel('Tasa de Supervivencia')
        ax.set_title('Tasa de Supervivencia por Clúster', fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'C{i}' for i in range(4)])
        ax.set_ylim(0,1.1)
        for i,v in enumerate(surv_by_cluster.values):
            ax.text(i, v+0.03, f'{v:.2%}', ha='center', fontweight='bold', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Perfil de clústeres
    st.subheader("👥 Perfiles de Clústeres")
    profile = df_c.groupby("Cluster")[["Pclass","Sex_enc","Age","Survived"]].mean().round(2)
    profile.columns = ["Pclass Promedio","Sex (0=F, 1=M)","Edad Promedio","Tasa Supervivencia"]
    profile["Tamaño"] = df_c["Cluster"].value_counts().sort_index()
    st.dataframe(profile, use_container_width=True)

    st.subheader("💬 Interpretación de Clústeres")
    for k, desc in CLUSTER_DESCRIPTIONS.items():
        color = COLORS[k]
        st.markdown(f"""
        <div class="cluster-card">
            <strong style="color:{color}">Clúster {k}</strong> — {desc}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PÁGINA 5 — MEJOR MODELO & CONCLUSIONES
# ══════════════════════════════════════════════
elif page == "🏆 Mejor Modelo & Conclusiones":
    st.title("🏆 Selección del Mejor Modelo & Conclusiones")

    st.subheader("🎯 Resumen de Decisiones")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📦 Mejor Conjunto de Características
        **X3 — Pclass, Sex, Age**
        - Silhouette: **0.5700** (máximo)
        - Davies-Bouldin: **0.6374** (mínimo)
        - 3 variables interpretables y clave
        """)
    with col2:
        st.markdown("""
        ### 🤖 Mejor Algoritmo
        **K-Means**
        - Convergencia rápida
        - Resultados consistentes
        - Mejores métricas vs configuraciones probadas
        """)
    with col3:
        st.markdown("""
        ### 🔢 Mejor K
        **K = 4**
        - Silhouette: 0.5700
        - Coincide con 4 grupos naturales del Titanic:
          Mujer/Hombre × 1ra clase/otras clases
        """)

    st.markdown("---")

    # Visualización final
    st.subheader("📊 Visualización Final — Clústeres en Espacio PCA")

    X_pca_c = best_clust["X_pca"]
    labels = best_clust["labels"]
    COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12']
    ev = pca_data["explained_ratio"]

    fig, axes = plt.subplots(1,2, figsize=(14,6))
    fig.suptitle("Mejor Modelo: K-Means K=4 sobre X3 (Pclass, Sex, Age)", fontsize=14, fontweight='bold')

    ax = axes[0]
    for k in range(4):
        mask = labels == k
        ax.scatter(X_pca_c[mask,0], X_pca_c[mask,1],
                   c=COLORS[k], label=CLUSTER_DESCRIPTIONS[k][:30]+'…',
                   alpha=0.7, s=55, edgecolors='k', linewidths=0.3)
    centers = best_clust["km"].cluster_centers_
    ax.scatter(centers[:,0], centers[:,1], c='black', s=200,
               marker='X', zorder=5, label='Centroides')
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)')
    ax.set_title('PC1 vs PC2 — Clústeres K-Means')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pclass_mean = df_c.groupby("Cluster")["Pclass"].mean()
    age_mean = df_c.groupby("Cluster")["Age"].mean()
    sex_mean = df_c.groupby("Cluster")["Sex_enc"].mean()
    surv_mean = df_c.groupby("Cluster")["Survived"].mean()

    width = 0.2
    x = np.arange(4)
    ax.bar(x - width, pclass_mean/3, width, label='Pclass (norm.)', color='#9b59b6', alpha=0.8)
    ax.bar(x, sex_mean, width, label='Sex (0=F,1=M)', color='#3498db', alpha=0.8)
    ax.bar(x + width, age_mean/80, width, label='Age (norm.)', color='#e67e22', alpha=0.8)
    ax.plot(x, surv_mean, 'r-o', linewidth=2, markersize=8, label='Surv. Rate', zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(4)])
    ax.set_title('Perfil Normalizado por Clúster')
    ax.set_ylabel('Valor Normalizado / Tasa')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Conclusiones
    st.markdown("---")
    st.subheader("📝 Conclusiones")
    st.markdown("""
    #### 🔍 Clasificación
    - **Random Forest** superó a Regresión Logística en todas las métricas (Acc: 83.2%, AUC: 0.894).
    - Las variables más predictivas son **Sex**, **FareLog** y **Age**, confirmando el patrón histórico.

    #### 📐 PCA
    - Con solo **5 componentes** se retiene el 85% de la varianza total.
    - En el espacio 2D (PC1 + PC2, 58.3% varianza), los grupos de supervivientes y no supervivientes presentan cierta separación pero con solapamiento.

    #### 🔵 Clustering
    - La mejor configuración es **X3 (Pclass, Sex, Age)** con **K=4**, Silhouette=0.57.
    - Los 4 clústeres corresponden naturalmente a: ①mujeres jóvenes económicas, ②hombres maduros premium, ③hombres jóvenes económicos, ④mujeres premium.
    - La tasa de supervivencia varía dramáticamente entre clústeres: del **16%** (C2, hombres 3ra clase) al **92%** (C3, mujeres 1ra clase).

    #### 🏛️ Contexto Histórico
    > *"Women and children first"* — El análisis cuantitativo confirma esta norma: el género y la clase social fueron los factores determinantes de supervivencia en el Titanic (1912).
    """)


# ══════════════════════════════════════════════
# PÁGINA 6 — PREDICTOR INDIVIDUAL
# ══════════════════════════════════════════════
elif page == "🔮 Predictor Individual":
    st.title("🔮 Predictor de Supervivencia Individual")
    st.markdown("Ingresa los datos de un pasajero y el modelo **Random Forest** predecirá si habría sobrevivido.")

    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("👤 Datos del Pasajero")
        pclass = st.selectbox("Clase (Pclass)", [1,2,3], index=2)
        sex = st.selectbox("Sexo", ["female","male"])
        age = st.slider("Edad", 1, 80, 30)
        fare = st.slider("Tarifa pagada (£)", 0, 500, 50)
        sibsp = st.number_input("Hermanos/cónyuge a bordo (SibSp)", 0, 8, 0)
        parch = st.number_input("Padres/hijos a bordo (Parch)", 0, 6, 0)
        embarked = st.selectbox("Puerto de embarque", ["S (Southampton)","C (Cherbourg)","Q (Queenstown)"])
        embarked_code = embarked[0]

    with col2:
        st.subheader("📊 Resultado de la Predicción")
        passenger = {
            "pclass": pclass, "sex": sex, "age": age, "fare": fare,
            "sibsp": sibsp, "parch": parch, "embarked": embarked_code
        }
        result = predict_passenger(rf, scaler, passenger)
        prob_surv = result["survived_prob"]
        prob_no = result["not_survived_prob"]

        if result["prediction"] == 1:
            st.success(f"✅ **HABRÍA SOBREVIVIDO** con probabilidad del {prob_surv:.1%}")
        else:
            st.error(f"❌ **NO HABRÍA SOBREVIVIDO** con probabilidad del {prob_no:.1%}")

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5,3))
        bars = ax.barh(['No Sobrevive','Sobrevive'],
                       [prob_no, prob_surv],
                       color=['#e74c3c','#2ecc71'], edgecolor='black', linewidth=0.8)
        ax.set_xlim(0,1)
        ax.set_xlabel('Probabilidad')
        ax.set_title('Probabilidades del Modelo RF', fontweight='bold')
        for bar, val in zip(bars, [prob_no, prob_surv]):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.1%}', va='center', fontweight='bold', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Factores contextuales
        st.markdown("**📌 Factores clave en tu caso:**")
        if sex == "female":
            st.markdown("🟢 Ser **mujer** aumenta significativamente las probabilidades")
        else:
            st.markdown("🔴 Ser **hombre** reduce significativamente las probabilidades")
        if pclass == 1:
            st.markdown("🟢 Viajar en **1ª clase** es favorable")
        elif pclass == 3:
            st.markdown("🔴 Viajar en **3ª clase** es desfavorable")
        if age < 16:
            st.markdown("🟢 Ser **niño** aumenta las probabilidades")
        if fare > 50:
            st.markdown("🟢 Tarifa **alta** es señal de mejor posición")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Proyecto IA - Titanic Dataset · Universidad Autonoma de Bucaramanga · 2026"
    "</div>",
    unsafe_allow_html=True
)
