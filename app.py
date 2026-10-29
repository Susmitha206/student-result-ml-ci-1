
from pathlib import Path
import logging
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("student-result-api")

# Prometheus metrics
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the model is loaded"
)

# Load the model once when the application starts
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    MODEL_LOADED.set(1)
    logger.info("Model loaded successfully")

except Exception:
    MODEL_LOADED.set(0)
    logger.exception("Model loading failed")
    raise


@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.perf_counter()

    logger.info(
        "Request started | request_id=%s method=%s path=%s",
        g.request_id, request.method, request.path
    )


@app.after_request
def after_request(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.perf_counter())
    duration = time.perf_counter() - start_time

    response.headers["X-Request-ID"] = request_id

    # Exclude metrics scraping from user-traffic metrics
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration)

    logger.info(
        "Request completed | request_id=%s status=%s duration_ms=%.2f",
        request_id, response.status_code, duration * 1000
    )

    return response


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    request_id = getattr(g, "request_id", "unknown")

    logger.exception(
        "Unexpected error | request_id=%s", request_id
    )

    return jsonify({
        "error": "Internal server error",
        "request_id": request_id
    }), 500


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning("Prediction rejected: JSON body is missing")
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "Prediction rejected: missing required fields=%s",
            missing_fields
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature] for feature in FEATURES
        }])

        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        PREDICTIONS.labels(result="success").inc()

        logger.info(
            "Prediction successful | request_id=%s prediction=%s",
            getattr(g, "request_id", "unknown"), prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception(
            "Prediction failed | request_id=%s",
            getattr(g, "request_id", "unknown")
        )
        raise


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
