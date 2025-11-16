# inference.py
import argparse
from pathlib import Path

import joblib
import pandas as pd


def load_pkl_folder(folder_path: str) -> pd.DataFrame:
    """Load and concatenate all .pkl files in a folder."""
    folder = Path(folder_path)
    files = sorted(folder.glob("*.pkl"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pkl files found in folder: {folder}")

    print(f"Found {len(files)} PKL files in {folder}. Loading...")
    dfs = [pd.read_pickle(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from PKL files.")
    return df


def load_any_data(path_str: str) -> pd.DataFrame:
    """
    Load data from:
      - a CSV file (.csv)
      - a single PKL file (.pkl)
      - a folder containing multiple .pkl files
    """
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")

    if path.is_dir():
        # folder of PKL files
        return load_pkl_folder(path_str)

    # file case
    suffix = path.suffix.lower()
    if suffix == ".csv":
        print(f"Loading CSV file: {path}")
        return pd.read_csv(path)
    elif suffix == ".pkl":
        print(f"Loading single PKL file: {path}")
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported data format: {suffix}. Use CSV or PKL (or a folder of PKLs).")


def preprocess_for_inference(df: pd.DataFrame, encoders: dict, features: list[str]):
    """Apply same preprocessing as training, without using TX_FRAUD."""
    df = df.copy()

    required_cols = ["TX_DATETIME", "TX_AMOUNT", "CUSTOMER_ID", "TERMINAL_ID"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    # time features
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    df["TX_DAY_OF_WEEK"] = df["TX_DATETIME"].dt.dayofweek
    df["TX_MONTH"] = df["TX_DATETIME"].dt.month

    # categorical encodings
    for col in ["CUSTOMER_ID", "TERMINAL_ID"]:
        if col not in encoders:
            raise KeyError(f"Encoder for {col} not found in artifact.")
        le = encoders[col]
        df[col] = le.transform(df[col].astype(str))

    X = df[features].values
    return X, df


def run_inference(model_path: str, data_path: str, out_path: str):
    print(f"Loading model artifact from: {model_path}")
    artifact = joblib.load(model_path)
    clf = artifact["model"]
    encoders = artifact["encoders"]
    feature_names = artifact["features"]

    print(f"Loading input data from: {data_path}")
    df = load_any_data(data_path)

    print("Preprocessing data...")
    X, df_proc = preprocess_for_inference(df, encoders, feature_names)

    print("Running predictions...")
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # keep original columns + predictions
    df_out = df.copy()
    df_out["FRAUD_PROB"] = probs
    df_out["PRED_FRAUD"] = preds

    out_path = Path(out_path)
    df_out.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Fraud detection inference (CSV or PKL)")
    parser.add_argument(
        "--data",
        "-d",
        required=True,
        help="Path to data: CSV file, single PKL file, or folder with PKL files",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="artifacts/fraud_model.joblib",
        help="Path to trained model artifact",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="predictions.csv",
        help="Output CSV file with predictions",
    )
    args = parser.parse_args()

    run_inference(args.model, args.data, args.out)


if __name__ == "__main__":
    main()

