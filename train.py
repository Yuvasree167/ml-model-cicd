from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from skl2onnx import to_onnx
from skl2onnx.common.data_types import (
    Int64TensorType,
    StringTensorType,
    DoubleTensorType,
)

DATA = Path("data/drug200.csv")
MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("Dataset missing. Run: python download_dataset.py")

df = pd.read_csv(DATA)
required = {"Age", "Sex", "BP", "Cholesterol", "Na_to_K", "Drug"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
df = df[list(required)].dropna()
X = df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]
y = df["Drug"]

categorical = ["Sex", "BP", "Cholesterol"]
numerical = ["Age", "Na_to_K"]
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numerical", "passthrough", numerical),
])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")

joblib.dump(pipeline, MODEL_DIR / "drug_model.joblib")

# Export with column-wise input types so browser inputs retain their original types.
initial_types = [
    ("Age", Int64TensorType([None, 1])),
    ("Sex", StringTensorType([None, 1])),
    ("BP", StringTensorType([None, 1])),
    ("Cholesterol", StringTensorType([None, 1])),
    ("Na_to_K", DoubleTensorType([None, 1])),
]
onnx_model = to_onnx(
    pipeline, initial_types=initial_types, target_opset=17,
    options={id(pipeline): {"zipmap": False}}
)
(MODEL_DIR / "drug_model.onnx").write_bytes(onnx_model.SerializeToString())

report = (
    f"Dataset rows used: {len(df)}\n"
    f"Training rows: {len(X_train)}\n"
    f"Test rows: {len(X_test)}\n"
    f"Accuracy: {accuracy:.4f}\n"
    f"Weighted F1-score: {f1:.4f}\n\n"
    f"{classification_report(y_test, predictions, zero_division=0)}"
)
(RESULTS_DIR / "metrics.txt").write_text(report, encoding="utf-8")
ConfusionMatrixDisplay.from_predictions(y_test, predictions, xticks_rotation="vertical")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()
(RESULTS_DIR / "summary.json").write_text(json.dumps({
    "rows": len(df), "accuracy": accuracy, "weighted_f1": f1,
    "classes": sorted(map(str, y.unique()))
}, indent=2), encoding="utf-8")
print(report)
print("Training, evaluation, and ONNX export completed.")
