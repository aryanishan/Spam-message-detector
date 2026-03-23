from io import BytesIO
import os
import pickle
import re
import math

from flask import Flask, jsonify, render_template, request

try:
    import pandas as pd
except ImportError:
    pd = None


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_ORIGINS = {
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

TEXT_COLUMN_CANDIDATES = [
    "message",
    "messages",
    "msg",
    "sms",
    "text",
    "content",
    "body",
    "v2",
]
LABEL_COLUMN_CANDIDATES = [
    "label",
    "labels",
    "category",
    "class",
    "target",
    "type",
    "spam",
    "v1",
]


def clean_text(text):
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(value):
    raw = "" if value is None else str(value).strip().lower()
    if raw in {"spam", "1", "true", "yes", "positive"}:
        return "SPAM"
    if raw in {"ham", "0", "false", "no", "negative"}:
        return "HAM"
    return None


def predict_message(message):
    cleaned = clean_text(message)
    if not cleaned:
        raise ValueError("Message cannot be empty.")

    vector = vectorizer.transform([cleaned])
    prediction_value = model.predict(vector)[0]
    label = normalize_label(prediction_value)
    if label is None:
        try:
            label = "SPAM" if int(prediction_value) == 1 else "HAM"
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported prediction label returned by model: {prediction_value}")

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

    return {
        "message": str(message),
        "cleaned_message": cleaned,
        "prediction": label,
        "confidence": confidence,
    }


def find_best_column(columns, candidates):
    lowered = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    for column in columns:
        column_name = str(column).strip().lower()
        if any(candidate in column_name for candidate in candidates):
            return column
    return None


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if value is None:
        return None
    if pd is not None and pd.isna(value):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_uploaded_dataframe(uploaded_file):
    if pd is None:
        raise ImportError(
            "Missing dependency: pandas is required for file analysis. "
            "Install pandas and openpyxl, then try again."
        )

    filename = uploaded_file.filename or ""
    extension = os.path.splitext(filename)[1].lower()
    payload = uploaded_file.read()
    uploaded_file.stream.seek(0)

    if not payload:
        raise ValueError("The uploaded file is empty.")

    buffer = BytesIO(payload)

    if extension == ".csv":
        dataframe = pd.read_csv(buffer)
    elif extension == ".xlsx":
        dataframe = pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")

    return dataframe, len(payload)


def analyze_dataframe(dataframe, filename, file_size):
    if dataframe.empty:
        raise ValueError("The uploaded file has no data rows to analyze.")

    text_column = find_best_column(dataframe.columns, TEXT_COLUMN_CANDIDATES)
    if text_column is None:
        available_columns = ", ".join(str(column) for column in dataframe.columns)
        raise ValueError(
            "Could not find a message column. Expected names like "
            "message, sms, text, content, or v2. "
            f"Available columns: {available_columns}"
        )

    label_column = find_best_column(dataframe.columns, LABEL_COLUMN_CANDIDATES)

    original_rows, original_columns = dataframe.shape
    original_column_names = [str(column) for column in dataframe.columns]
    original_missing_values = {
        str(column): int(count) for column, count in dataframe.isna().sum().items()
    }

    working_df = dataframe.copy()
    working_df[text_column] = working_df[text_column].fillna("").astype(str)
    working_df["predicted_label"] = working_df[text_column].apply(
        lambda item: predict_message(item)["prediction"] if clean_text(item) else "EMPTY"
    )

    spam_count = int((working_df["predicted_label"] == "SPAM").sum())
    ham_count = int((working_df["predicted_label"] == "HAM").sum())
    empty_count = int((working_df["predicted_label"] == "EMPTY").sum())

    response = {
        "file_name": filename,
        "file_size_kb": round(file_size / 1024, 2),
        "rows": int(original_rows),
        "columns": int(original_columns),
        "column_names": original_column_names,
        "message_column": str(text_column),
        "label_column": str(label_column) if label_column is not None else None,
        "predicted_counts": {
            "spam": spam_count,
            "ham": ham_count,
            "empty": empty_count,
        },
        "missing_values": original_missing_values,
        "preview": sanitize_for_json(working_df.head(8).to_dict(orient="records")),
    }

    if label_column is not None:
        actual_labels = working_df[label_column].apply(normalize_label)
        valid_label_rows = int(actual_labels.isin(["SPAM", "HAM"]).sum())
        if valid_label_rows == 0:
            raise ValueError(
                "Invalid dataset: the detected label column does not contain valid spam/ham values."
            )
        comparable_mask = actual_labels.isin(["SPAM", "HAM"]) & working_df["predicted_label"].isin(
            ["SPAM", "HAM"]
        )
        comparable_rows = int(comparable_mask.sum())

        actual_spam_count = int((actual_labels == "SPAM").sum())
        actual_ham_count = int((actual_labels == "HAM").sum())

        response["actual_counts"] = {
            "spam": actual_spam_count,
            "ham": actual_ham_count,
        }

        if comparable_rows > 0:
            matches = int((actual_labels[comparable_mask] == working_df.loc[comparable_mask, "predicted_label"]).sum())
            response["evaluation"] = {
                "comparable_rows": comparable_rows,
                "matched_rows": matches,
                "accuracy_percent": round((matches / comparable_rows) * 100, 2),
            }

    return sanitize_for_json(response)


@app.route("/")
def home():
    return render_template("index.html")


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/predict", methods=["OPTIONS"])
@app.route("/api/analyze-file", methods=["OPTIONS"])
def handle_preflight():
    return ("", 204)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")

    try:
        result = predict_message(message)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/api/analyze-file", methods=["POST"])
def analyze_file():
    uploaded_file = request.files.get("file")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please upload a CSV or Excel file."}), 400

    try:
        dataframe, file_size = load_uploaded_dataframe(uploaded_file)
        analysis = analyze_dataframe(dataframe, uploaded_file.filename, file_size)
        return jsonify(analysis)
    except (ValueError, ImportError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"File analysis failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
