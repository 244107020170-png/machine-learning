import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report


# STEP 1 - LOAD DATA

df = pd.read_csv("Titanic-Dataset.csv")

print("=== DATASET INFO ===")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# STEP 2 - DATA GROUPING

# Target variable
y = df["Survived"].astype(int)

# Features
X = df.drop(columns=["Survived"])

# Numerical features
num_cols = ["Age", "SibSp", "Parch", "Fare"]

# Categorical features
cat_cols = ["Pclass", "Sex", "Embarked"]


# STEP 3 - FEATURE EXTRACTION

# Numerical transformation
num_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical transformation
cat_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# STEP 3.1 - FEATURE CONSTRUCTION

# FamilySize = SibSp + Parch + 1
X["FamilySize"] = (
    X["SibSp"].fillna(0)
    + X["Parch"].fillna(0)
    + 1
)

# Add FamilySize to numerical features
preprocess = ColumnTransformer([
    ("num", num_tf, num_cols + ["FamilySize"]),
    ("cat", cat_tf, cat_cols),
])


# STEP 4 - FEATURE SELECTION

# Select 5 best features using ANOVA
selector_filter = SelectKBest(
    score_func=f_classif,
    k=5
)


# Final pipeline
pipe_filter = Pipeline([
    ("prep", preprocess),
    ("sel", selector_filter),
    ("clf", LogisticRegression(max_iter=1000))
])


# STEP 5 - TRAIN AND TEST MODEL

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

# Train
pipe_filter.fit(X_train, y_train)

# Prediction
pred = pipe_filter.predict(X_test)


# EVALUATION

print("\n")
print("=== Filter (ANOVA) + Logistic Regression ===")

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, pred))


# FEATURE INSPECTION

# Feature names after preprocessing
feat_names = pipe_filter.named_steps[
    "prep"
].get_feature_names_out()

print("\n")
print("=== FEATURE NAMES AFTER PREPROCESSING ===")
print(feat_names)


# Get selected features
sel = pipe_filter.named_steps["sel"]

mask = sel.get_support()

selected_names = feat_names[mask]
selected_scores = sel.scores_[mask]


# Sort selected features by ANOVA score
top = sorted(
    zip(selected_names, selected_scores),
    key=lambda t: t[1],
    reverse=True
)

print("\n=== SELECTED FEATURES ===")

for feature, score in top:
    print(f"{feature}: {score:.4f}")


# EXPERIMENT - DIFFERENT VALUES OF K

print("\n")
print("=" * 60)
print("EXPERIMENT: DIFFERENT VALUES OF K")
print("=" * 60)

# Total features after preprocessing = 13
k_values = [3, 5, 7, 9, 11, 13]

results = []

for k in k_values:

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    pipeline = Pipeline([
        ("prep", preprocess),
        ("sel", selector),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    results.append({
        "k": k,
        "accuracy": acc
    })

    print(f"k = {k:2d} | Accuracy = {acc:.4f}")


# FIND BEST K

results_df = pd.DataFrame(results)

best_result = results_df.loc[
    results_df["accuracy"].idxmax()
]

print("\n")
print("=== BEST CONFIGURATION ===")

print(
    f"Best k = {int(best_result['k'])}"
)

print(
    f"Best Accuracy = {best_result['accuracy']:.4f}"
)


# SHOW FEATURES OF BEST K

best_k = int(best_result["k"])

best_selector = SelectKBest(
    score_func=f_classif,
    k=best_k
)

best_pipeline = Pipeline([
    ("prep", preprocess),
    ("sel", best_selector),
    ("clf", LogisticRegression(max_iter=1000))
])

best_pipeline.fit(X_train, y_train)

best_feat_names = best_pipeline.named_steps[
    "prep"
].get_feature_names_out()

best_sel = best_pipeline.named_steps["sel"]

best_mask = best_sel.get_support()

best_features = best_feat_names[best_mask]

print("\n")
print("=== FEATURES USED BY BEST MODEL ===")

for i, feature in enumerate(best_features, start=1):
    print(f"{i}. {feature}")