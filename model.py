"""
model.py — Lógica de ML para el análisis del Titanic
Carga datos, limpia, entrena clasificadores, aplica PCA y clustering.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, silhouette_score, davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

FEATURES = ["Pclass", "Sex_enc", "Age", "SibSp", "Parch", "FareLog", "Embarked_enc", "FamilySize"]
FEATURE_LABELS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "FareLog", "Embarked", "FamilySize"]

CLUSTER_DESCRIPTIONS = {
    0: "👩 Mujeres jóvenes (2da/3ra clase) — supervivencia media (59%)",
    1: "👨 Hombres mayores (1ra clase) — baja supervivencia (26%)",
    2: "👨 Hombres jóvenes (2da/3ra clase) — muy baja supervivencia (16%)",
    3: "👩 Mujeres adultas (1ra clase) — alta supervivencia (92%)",
}


# ──────────────────────────────────────────────
# 1. CARGA Y LIMPIEZA
# ──────────────────────────────────────────────

def load_and_clean(url: str = TITANIC_URL) -> pd.DataFrame:
    """Descarga y limpia el dataset del Titanic."""
    df = pd.read_csv(url)

    # Valores faltantes
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Encoding
    le = LabelEncoder()
    df["Sex_enc"] = le.fit_transform(df["Sex"])          # female=0, male=1
    df["Embarked_enc"] = le.fit_transform(df["Embarked"])

    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FareLog"] = np.log1p(df["Fare"])

    return df


# ──────────────────────────────────────────────
# 2. CLASIFICACIÓN
# ──────────────────────────────────────────────

def train_classifiers(df: pd.DataFrame):
    """Entrena Logistic Regression y Random Forest. Devuelve métricas y objetos."""
    X = df[FEATURES].copy()
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_prob = lr.predict_proba(X_test_sc)[:, 1]
    lr_cv = cross_val_score(lr, scaler.transform(X), y, cv=5, scoring="accuracy").mean()

    # ── Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean()

    metrics = {
        "lr": {
            "accuracy": accuracy_score(y_test, lr_pred),
            "auc": roc_auc_score(y_test, lr_prob),
            "cv_accuracy": lr_cv,
            "cm": confusion_matrix(y_test, lr_pred),
            "report": classification_report(y_test, lr_pred, output_dict=True),
        },
        "rf": {
            "accuracy": accuracy_score(y_test, rf_pred),
            "auc": roc_auc_score(y_test, rf_prob),
            "cv_accuracy": rf_cv,
            "cm": confusion_matrix(y_test, rf_pred),
            "report": classification_report(y_test, rf_pred, output_dict=True),
            "feature_importances": pd.Series(
                rf.feature_importances_, index=FEATURE_LABELS
            ).sort_values(ascending=False),
        },
        "y_test": y_test,
    }

    winner = "rf" if metrics["rf"]["auc"] > metrics["lr"]["auc"] else "lr"
    metrics["winner"] = winner

    return lr, rf, scaler, metrics, X_test, X_test_sc, y_test


# ──────────────────────────────────────────────
# 3. PCA
# ──────────────────────────────────────────────

def run_pca(df: pd.DataFrame):
    """Aplica PCA completo y retorna componentes, varianza y proyección 2D."""
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n85 = int(np.argmax(cum_var >= 0.85)) + 1

    pca2 = PCA(n_components=2, random_state=42)
    X_pca = pca2.fit_transform(X_scaled)

    return {
        "scaler": scaler,
        "pca_full": pca_full,
        "pca2": pca2,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "explained_ratio": pca_full.explained_variance_ratio_,
        "cum_var": cum_var,
        "n_components_85": n85,
    }


# ──────────────────────────────────────────────
# 4. CLUSTERING  (K-Means sobre los 3 conjuntos)
# ──────────────────────────────────────────────

FEATURE_SETS = {
    "X1 — FareLog, Age, Sex": ["FareLog", "Age", "Sex_enc"],
    "X2 — Todas (8 variables)": FEATURES,
    "X3 — Pclass, Sex, Age ⭐": ["Pclass", "Sex_enc", "Age"],
}


def run_clustering(df: pd.DataFrame):
    """Ejecuta K-Means con K=2,3,4 sobre los 3 conjuntos de características."""
    results = []

    for fname, cols in FEATURE_SETS.items():
        Xs = StandardScaler().fit_transform(df[cols])
        Xp = PCA(n_components=2, random_state=42).fit_transform(Xs)
        for k in [2, 3, 4]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(Xp)
            sil = silhouette_score(Xp, lbl)
            db = davies_bouldin_score(Xp, lbl)
            results.append({
                "Conjunto": fname,
                "K": k,
                "Silhouette ↑": round(sil, 4),
                "Davies-Bouldin ↓": round(db, 4),
            })

    return pd.DataFrame(results).sort_values("Silhouette ↑", ascending=False).reset_index(drop=True)


def get_best_clustering(df: pd.DataFrame):
    """Retorna el mejor modelo: X3, K-Means, K=4."""
    cols = ["Pclass", "Sex_enc", "Age"]
    Xs = StandardScaler().fit_transform(df[cols])
    pca2 = PCA(n_components=2, random_state=42)
    Xp = pca2.fit_transform(Xs)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(Xp)
    sil = silhouette_score(Xp, labels)
    db = davies_bouldin_score(Xp, labels)
    return {
        "labels": labels,
        "X_pca": Xp,
        "km": km,
        "silhouette": sil,
        "db": db,
        "cluster_descriptions": CLUSTER_DESCRIPTIONS,
    }


# ──────────────────────────────────────────────
# 5. PREDICCIÓN INDIVIDUAL
# ──────────────────────────────────────────────

def predict_passenger(rf_model, scaler, passenger: dict) -> dict:
    """Predice supervivencia de un pasajero nuevo con Random Forest."""
    row = pd.DataFrame([{
        "Pclass": passenger["pclass"],
        "Sex_enc": 1 if passenger["sex"] == "male" else 0,
        "Age": passenger["age"],
        "SibSp": passenger["sibsp"],
        "Parch": passenger["parch"],
        "FareLog": np.log1p(passenger["fare"]),
        "Embarked_enc": {"S": 2, "C": 0, "Q": 1}.get(passenger["embarked"], 2),
        "FamilySize": passenger["sibsp"] + passenger["parch"] + 1,
    }])
    prob = rf_model.predict_proba(row)[0]
    return {
        "survived_prob": prob[1],
        "not_survived_prob": prob[0],
        "prediction": int(prob[1] >= 0.5),
    }
