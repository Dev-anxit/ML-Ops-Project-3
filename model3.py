import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_and_preprocess_data(filepath="Student_Performance_DT.csv"):
    """Load and preprocess the student performance dataset."""
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    print("\nDataset Preview:")
    print(df.head())
    print("\nMissing Values Count:")
    print(df.isna().sum())

    # Map binary categorical columns to numeric
    df["Internet_Access"] = df["Internet_Access"].map({"Yes": 1, "No": 0})
    df["Extracurricular"] = df["Extracurricular"].map({"Yes": 1, "No": 0})
    df["Final_Result"] = df["Final_Result"].map({"Pass": 1, "Fail": 0})

    feature_cols = [
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Assignments_Completed",
        "Sleep_Hours",
        "Participation",
        "Internet_Access",
        "Extracurricular",
    ]
    X = df[feature_cols]
    y = df["Final_Result"]

    print(f"\nFeatures shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def train_and_evaluate(X, y):
    """Train baseline and hyperparameter-tuned Decision Tree models."""
    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Baseline Decision Tree Model
    print("\n" + "=" * 50)
    print("Training Baseline Decision Tree Classifier...")
    print("=" * 50)
    trad_model = DecisionTreeClassifier(random_state=42)
    trad_model.fit(X_train, y_train)

    y_pred_trad = trad_model.predict(X_test)
    trad_acc = accuracy_score(y_test, y_pred_trad)
    print(f"Baseline Test Accuracy: {trad_acc:.4f}")
    print("\nBaseline Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_trad))
    print("\nBaseline Classification Report:")
    print(classification_report(y_test, y_pred_trad))

    # 2. Hyperparameter Tuning using GridSearchCV
    print("=" * 50)
    print("Hyperparameter Tuning with GridSearchCV...")
    print("=" * 50)
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5],
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    # Evaluate Tuned Model
    tuned_model = grid_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)
    tuned_acc = accuracy_score(y_test, y_pred_tuned)

    print(f"\nTuned Model Test Accuracy: {tuned_acc:.4f}")
    print("\nTuned Model Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_tuned))
    print("\nTuned Model Classification Report:")
    print(classification_report(y_test, y_pred_tuned))


def main():
    X, y = load_and_preprocess_data()
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()
