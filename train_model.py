"""
Titanic Survival Prediction - Model Training Script
-----------------------------------------------------
Loads the Titanic dataset, engineers features, trains and compares
multiple models, then saves the best one (+ preprocessing objects)
for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("data/train.csv")
print(f"Loaded {len(df)} rows")

# -----------------------------
# 2. Feature engineering
# -----------------------------

# Extract title from name (Mr, Mrs, Miss, Master, Rare)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
               "Rev", "Sir", "Jonkheer", "Dona"]
df["Title"] = df["Title"].replace(rare_titles, "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# Family size features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Fill missing Age with median per Title group (more accurate than overall median)
df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill missing Fare with median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Cabin: just capture whether it was known (missing cabin info was correlated with class/survival)
df["HasCabin"] = df["Cabin"].notna().astype(int)

# -----------------------------
# 3. Encode categoricals
# -----------------------------
le_sex = LabelEncoder()
df["Sex_enc"] = le_sex.fit_transform(df["Sex"])  # female=0, male=1

le_embarked = LabelEncoder()
df["Embarked_enc"] = le_embarked.fit_transform(df["Embarked"])

le_title = LabelEncoder()
df["Title_enc"] = le_title.fit_transform(df["Title"])

# -----------------------------
# 4. Select features
# -----------------------------
features = ["Pclass", "Sex_enc", "Age", "Fare", "FamilySize",
            "IsAlone", "HasCabin", "Embarked_enc", "Title_enc"]

X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale (mainly benefits Logistic Regression; harmless for tree models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Train & compare models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = (model, acc, preds)
    print(f"\n{name}: accuracy = {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Died", "Survived"]))

# -----------------------------
# 6. Pick best model
# -----------------------------
best_name = max(results, key=lambda k: results[k][1])
best_model, best_acc, best_preds = results[best_name]
print(f"\n>>> Best model: {best_name} ({best_acc:.4f} accuracy)")

print("\nConfusion Matrix (best model):")
print(confusion_matrix(y_test, best_preds))

# Feature importance (if available)
if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=features)
    importance = importance.sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance)
    importance.to_csv("feature_importance.csv")

# -----------------------------
# 7. Save everything the app needs
# -----------------------------
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump({
        "sex": le_sex,
        "embarked": le_embarked,
        "title": le_title,
        "model_name": best_name,
        "accuracy": best_acc,
        "uses_scaling": best_name == "Logistic Regression",
        "features": features,
    }, f)

print("\nSaved: titanic_model.pkl, scaler.pkl, encoders.pkl")
