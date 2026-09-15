# LAB ASSIGNMENT
# Wisconsin Breast Cancer
# Feature Selection + Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# STEP 1 - LOAD DATASET

df = pd.read_csv("wbc.csv")

print("=== DATASET INFO ===")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# STEP 2 - DIFFERENTIATE USABLE AND
#         UNUSABLE VARIABLES

target = "diagnosis"

# 'id' is only an identifier.
# 'Unnamed: 32' is an empty column containing only NaN.
unusable = ["id", "Unnamed: 32"]

# Remove unusable columns and target from X
X = df.drop(columns=[target] + unusable)

# Encode diagnosis:
# M = 1 (Malignant)
# B = 0 (Benign)
y = df[target].map({"M": 1, "B": 0})


print("\n=== VARIABLE SELECTION ===")
print("Unusable variables:", unusable)
print("Number of usable features:", X.shape[1])


# STEP 3 - CHECK NUMERICAL FEATURES

num_cols = X.select_dtypes(include="number").columns.tolist()

print("\n=== NUMERICAL FEATURES ===")
print("Number of numerical features:", len(num_cols))
print(num_cols)


# STEP 4 - TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# STEP 5 - FEATURE SELECTION
#         + STANDARDIZATION
#         + LOGISTIC REGRESSION

k_values = [5, 10, 15, 20, 25, 30]

results = []

print("\n")
print("=" * 60)
print("EXPERIMENT: DIFFERENT VALUES OF K")
print("=" * 60)

for k in k_values:

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append((k, accuracy))

    print(f"k = {k:2d} | Accuracy = {accuracy:.4f}")


# STEP 6 - DETERMINE BEST K

best_k, best_accuracy = max(results, key=lambda x: x[1])

print("\n")
print("=== BEST CONFIGURATION ===")
print("Best k:", best_k)
print("Best Accuracy:", f"{best_accuracy:.4f}")


# STEP 7 - TRAIN BEST MODEL

best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(score_func=f_classif, k=best_k)),
    ("classifier", LogisticRegression(max_iter=1000))
])

best_pipeline.fit(X_train, y_train)

best_pred = best_pipeline.predict(X_test)


# STEP 8 - EVALUATION

print("\n")
print("=== BEST MODEL EVALUATION ===")
print("Accuracy:", accuracy_score(y_test, best_pred))

print("\nClassification Report:")
print(classification_report(y_test, best_pred))


# STEP 9 - SHOW SELECTED FEATURES

selector = best_pipeline.named_steps["selector"]

selected_mask = selector.get_support()

selected_features = X.columns[selected_mask]

print("\n")
print("=== OPTIMAL FEATURES ===")
print("Number of optimal features:", len(selected_features))

for i, feature in enumerate(selected_features, start=1):
    print(f"{i}. {feature}")


# STEP 10 - SHOW FEATURE SCORES

scores = selector.scores_

feature_scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": scores
})

feature_scores = feature_scores.sort_values(
    by="Score",
    ascending=False
)

print("\n")
print("=== FEATURE SCORES ===")
print(feature_scores.to_string(index=False))