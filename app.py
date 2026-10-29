
from pathlib import Path
import logging
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

logger = logging.getLogger("student-result-api")


# Track each request and its duration
@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()

    logger.info(
        "Request started | request_id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path
    )


@app.after_request
def after_request(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.time())
    duration_ms = (time.time() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request completed | request_id=%s status=%s duration_ms=%.2f",
        request_id,
        response.status_code,
        duration_ms
    )

    return response


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    request_id = getattr(g, "request_id", "unknown")

    logger.exception(
        "Unexpected error | request_id=%s",
        request_id
    )

    return jsonify({
        "error": "Internal server error",
        "request_id": request_id
    }), 500


def load_model():
    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy"
    }), 200


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning("Prediction rejected: JSON body is missing")
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
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
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        logger.info(
            "Prediction successful | request_id=%s prediction=%s",
            getattr(g, "request_id", "unknown"),
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except Exception:
        logger.exception(
            "Prediction failed | request_id=%s",
            getattr(g, "request_id", "unknown")
        )
        raise


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
