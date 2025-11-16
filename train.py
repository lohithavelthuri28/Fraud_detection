# train.py - PKL version
import argparse
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_pkl_folder(folder_path: str) -> pd.DataFrame:
    folder = Path(folder_path)
    files = sorted(folder.glob("*.pkl"))

    if len(files) == 0:
        raise FileNotFoundError(f"No .pkl files found in folder: {folder_path}")

    print(f"Found {len(files)} PKL files. Loading...")

    dfs = []
    for f in files:
        dfs.append(pd.read_pickle(f))

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} total rows.")
    return df


def preprocess(df: pd.DataFrame, fit_encoders=True, encoders=None):
    df = df.copy()
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    df["TX_DAY_OF_WEEK"] = df["TX_DATETIME"].dt.dayofweek
    df["TX_MONTH"] = df["TX_DATETIME"].dt.month

    features = ["TX_AMOUNT", "TX_HOUR", "TX_DAY_OF_WEEK", "TX_MONTH"]

    if encoders is None:
        encoders = {}

    for col in ["CUSTOMER_ID", "TERMINAL_ID"]:
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            df[col] = encoders[col].transform(df[col].astype(str))
        features.append(col)

    X = df[features].values
    y = df["TX_FRAUD"].values
    return X, y, encoders, features


def train_model(folder_path: str, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)

    df = load_pkl_folder(folder_path)
    print(f"Fraud cases: {df['TX_FRAUD'].sum()}")

    X, y, encoders, feature_names = preprocess(df, fit_encoders=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForest model...")
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    model_path = Path(out_dir) / "fraud_model.joblib"
    joblib.dump({"model": clf, "encoders": encoders, "features": feature_names}, model_path)
    print(f"\nModel saved to: {model_path.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", required=True, help="Folder containing PKL files")
    args = parser.parse_args()
    train_model(args.data)


if __name__ == "__main__":
    main()
