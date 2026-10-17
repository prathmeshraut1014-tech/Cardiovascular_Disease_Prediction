"""
Project 1: Cardiovascular Disease Prediction
--------------------------------------------
- Data Preprocessing
- Exploratory Data Analysis & Visualizations
- Correlation Matrix
- Model Comparison: SVM, KNN, Decision Tree, Logistic Regression, Random Forest
- Final Model for Heart Disease Detection
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cardio_train.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_and_preprocess():
    print("=" * 60)
    print("1. LOADING & PREPROCESSING DATA")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH, sep=";")
    print(f"Original shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")

    # Drop id column
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Age is in days → convert to years
    df["age"] = (df["age"] / 365).round().astype(int)

    # BMI feature
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    # Remove unrealistic values (outliers)
    # Blood pressure: ap_hi (systolic) 90-250, ap_lo (diastolic) 40-150
    # Height 120-220 cm, weight 40-200 kg, age 30-80
    before = len(df)
    df = df[
        (df["ap_hi"] >= 90) & (df["ap_hi"] <= 250) &
        (df["ap_lo"] >= 40) & (df["ap_lo"] <= 150) &
        (df["ap_hi"] > df["ap_lo"]) &
        (df["height"] >= 120) & (df["height"] <= 220) &
        (df["weight"] >= 40) & (df["weight"] <= 200) &
        (df["age"] >= 30) & (df["age"] <= 80) &
        (df["bmi"] >= 15) & (df["bmi"] <= 50)
    ].copy()
    print(f"\nRows removed as outliers: {before - len(df)}")
    print(f"Cleaned shape: {df.shape}")

    # Target distribution
    print(f"\nTarget (cardio) distribution:\n{df['cardio'].value_counts(normalize=True)}")

    return df


def exploratory_analysis(df):
    print("\n" + "=" * 60)
    print("2. EXPLORATORY DATA ANALYSIS & VISUALIZATIONS")
    print("=" * 60)

    # Target distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x="cardio", data=df, palette="Set2")
    plt.title("Distribution of Cardiovascular Disease (Target)")
    plt.xlabel("Cardio (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_target_distribution.png"), dpi=150)
    plt.close()
    print("Saved: 01_target_distribution.png")

    # Age distribution by cardio
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="age", hue="cardio", bins=30, kde=True, palette="Set1")
    plt.title("Age Distribution by Cardiovascular Disease")
    plt.xlabel("Age (years)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_age_distribution.png"), dpi=150)
    plt.close()
    print("Saved: 02_age_distribution.png")

    # Gender vs Cardio
    plt.figure(figsize=(7, 5))
    sns.countplot(x="gender", hue="cardio", data=df, palette="pastel")
    plt.title("Gender vs Cardiovascular Disease")
    plt.xlabel("Gender (1 = Female, 2 = Male)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_gender_vs_cardio.png"), dpi=150)
    plt.close()
    print("Saved: 03_gender_vs_cardio.png")

    # Cholesterol & Glucose
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x="cholesterol", hue="cardio", data=df, ax=axes[0], palette="muted")
    axes[0].set_title("Cholesterol Level vs Cardio")
    axes[0].set_xlabel("Cholesterol (1=normal, 2=above, 3=well above)")
    sns.countplot(x="gluc", hue="cardio", data=df, ax=axes[1], palette="muted")
    axes[1].set_title("Glucose Level vs Cardio")
    axes[1].set_xlabel("Glucose (1=normal, 2=above, 3=well above)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_cholesterol_glucose.png"), dpi=150)
    plt.close()
    print("Saved: 04_cholesterol_glucose.png")

    # Blood pressure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x="cardio", y="ap_hi", data=df, ax=axes[0], palette="Set3")
    axes[0].set_title("Systolic BP (ap_hi) by Cardio")
    sns.boxplot(x="cardio", y="ap_lo", data=df, ax=axes[1], palette="Set3")
    axes[1].set_title("Diastolic BP (ap_lo) by Cardio")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_blood_pressure.png"), dpi=150)
    plt.close()
    print("Saved: 05_blood_pressure.png")

    # BMI
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="cardio", y="bmi", data=df, palette="coolwarm")
    plt.title("BMI by Cardiovascular Disease")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_bmi_boxplot.png"), dpi=150)
    plt.close()
    print("Saved: 06_bmi_boxplot.png")

    # Lifestyle factors
    lifestyle = ["smoke", "alco", "active"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, col in enumerate(lifestyle):
        sns.countplot(x=col, hue="cardio", data=df, ax=axes[i], palette="Set2")
        axes[i].set_title(f"{col.capitalize()} vs Cardio")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "07_lifestyle_factors.png"), dpi=150)
    plt.close()
    print("Saved: 07_lifestyle_factors.png")

    # Pairplot of key numeric features (sample for speed)
    sample = df.sample(n=min(3000, len(df)), random_state=42)
    sns.pairplot(sample[["age", "bmi", "ap_hi", "ap_lo", "cardio"]],
                 hue="cardio", diag_kind="kde", palette="husl", corner=True)
    plt.suptitle("Pairplot of Key Features", y=1.02)
    plt.savefig(os.path.join(PLOTS_DIR, "08_pairplot.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: 08_pairplot.png")

    print("\nKey Insights:")
    print("- Higher age, higher BMI, elevated cholesterol/glucose and high BP are associated with higher cardio risk.")
    print("- Active lifestyle appears protective; smoking/alcohol effects are milder in this dataset.")


def correlation_matrix(df):
    print("\n" + "=" * 60)
    print("3. CORRELATION MATRIX")
    print("=" * 60)

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "09_correlation_matrix.png"), dpi=150)
    plt.close()
    print("Saved: 09_correlation_matrix.png")

    # Correlation with target
    target_corr = corr["cardio"].drop("cardio").sort_values(ascending=False)
    print("\nCorrelation with target (cardio):")
    print(target_corr)
    return corr


def train_and_compare_models(df):
    print("\n" + "=" * 60)
    print("4. MODEL TRAINING & COMPARISON")
    print("=" * 60)

    # Features & target
    feature_cols = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                    "cholesterol", "gluc", "smoke", "alco", "active", "bmi"]
    X = df[feature_cols]
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (important for SVM, KNN, LR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    # For speed: use a stratified sample for SVM (full data still used for other models)
    from sklearn.model_selection import train_test_split as tts
    X_svm, _, y_svm, _ = tts(X_train_scaled, y_train, train_size=12000, stratify=y_train, random_state=42)

    models = {
        "Logistic Regression (LR)": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "Support Vector Machine (SVM)": SVC(kernel="rbf", probability=True, random_state=42, cache_size=500),
        "Decision Tree (DT)": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest (RF)": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Use scaled data for LR, KNN, SVM; original for tree-based
        if name in ["Decision Tree (DT)", "Random Forest (RF)"]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        elif name == "Support Vector Machine (SVM)":
            model.fit(X_svm, y_svm)          # faster training on sample
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results.append({"Model": name, "Accuracy": acc, "ROC-AUC": auc})
        trained_models[name] = model
        print(f"  Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    print("\n" + "-" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("-" * 50)
    print(results_df.to_string(index=False))

    # Accuracy comparison plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.xlim(0.5, 0.8)
    for i, row in results_df.iterrows():
        plt.text(row["Accuracy"] + 0.005, i, f"{row['Accuracy']:.4f}", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "10_model_accuracy_comparison.png"), dpi=150)
    plt.close()
    print("\nSaved: 10_model_accuracy_comparison.png")

    # ROC curves
    plt.figure(figsize=(9, 7))
    for name, model in trained_models.items():
        if name in ["Decision Tree (DT)", "Random Forest (RF)"]:
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "11_roc_curves.png"), dpi=150)
    plt.close()
    print("Saved: 11_roc_curves.png")

    # Best model
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\nBest model by accuracy: {best_name}")

    # Confusion matrix of best model
    if best_name in ["Decision Tree (DT)", "Random Forest (RF)"]:
        y_pred_best = best_model.predict(X_test)
    else:
        y_pred_best = best_model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "12_confusion_matrix_best.png"), dpi=150)
    plt.close()
    print("Saved: 12_confusion_matrix_best.png")

    # Feature importance (for tree-based)
    if best_name in ["Decision Tree (DT)", "Random Forest (RF)"]:
        importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        plt.figure(figsize=(9, 5))
        sns.barplot(x=importances.values, y=importances.index, palette="magma")
        plt.title(f"Feature Importance - {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "13_feature_importance.png"), dpi=150)
        plt.close()
        print("Saved: 13_feature_importance.png")
        print("\nFeature Importances:")
        print(importances)

    # Save best model
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to: {model_path}")

    # Also save feature list
    with open(os.path.join(MODELS_DIR, "feature_columns.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    return results_df, best_name, best_model, scaler, feature_cols, X_test, y_test


def main():
    print("\n" + "#" * 60)
    print("  CARDIOVASCULAR DISEASE PREDICTION PROJECT")
    print("#" * 60)

    df = load_and_preprocess()
    exploratory_analysis(df)
    correlation_matrix(df)
    results_df, best_name, best_model, scaler, features, X_test, y_test = train_and_compare_models(df)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Best performing model : {best_name}")
    print(f"Plots saved in        : {PLOTS_DIR}")
    print(f"Model saved in        : {MODELS_DIR}")
    print("\nYou can use the saved model for predictions.")


if __name__ == "__main__":
    main()
