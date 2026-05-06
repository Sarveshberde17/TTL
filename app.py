import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from threading import Lock

import joblib
import pandas as pd
from flask import Flask, flash, g, jsonify, render_template, request
from redis import Redis
from redis.exceptions import RedisError
import sentry_sdk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "Cleaned_data.csv")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel.pkl")

METRICS_LOCK = Lock()
METRICS = {
    "total_requests": 0,
    "prediction_requests": 0,
    "error_responses": 0,
}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("house_price_app")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_secret_key(config: dict):
    aws_enabled = str(config.get("AWS_SECRETS_ENABLED", "false")).lower() == "true"
    if not aws_enabled:
        return os.getenv("SECRET_KEY")

    secret_name = config.get("AWS_SECRET_NAME")
    region_name = config.get("AWS_REGION")
    if not secret_name or not region_name:
        raise RuntimeError("AWS secrets enabled but AWS_SECRET_NAME/AWS_REGION are not set")

    try:
        import boto3  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("boto3 is required for AWS Secrets Manager integration") from exc

    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    secret_string = response.get("SecretString")
    if not secret_string:
        raise RuntimeError("AWS Secrets Manager returned empty SecretString")
    try:
        parsed = json.loads(secret_string)
        secret_key = parsed.get("SECRET_KEY")
        if secret_key:
            return secret_key
    except json.JSONDecodeError:
        return secret_string
    raise RuntimeError("SECRET_KEY not present in AWS secret payload")


def api_error(message: str, status_code: int = 400):
    return jsonify({"status": "error", "message": message}), status_code


def api_success(payload: dict, status_code: int = 200):
    response = {"status": "ok"}
    response.update(payload)
    return jsonify(response), status_code


def validate_model_artifact(model_path: str):
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"
    if os.path.getsize(model_path) == 0:
        return False, f"Model file is empty: {model_path}"
    return True, ""


def load_model(model_path: str, logger: logging.Logger):
    is_valid, validation_error = validate_model_artifact(model_path)
    if not is_valid:
        logger.error(validation_error)
        return None, validation_error
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model, None
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.exception("Model loading failed")
        return None, str(exc)


def load_dataset(data_path: str):
    return pd.read_csv(data_path)


def init_redis_client(redis_url: str, logger: logging.Logger):
    if not redis_url:
        return None
    try:
        client = Redis.from_url(redis_url, decode_responses=True)
        client.ping()
        logger.info("Redis rate limiter enabled")
        return client
    except RedisError:
        logger.warning("Redis unavailable, rate limiting disabled", exc_info=True)
        return None


def increment_metric(name: str):
    with METRICS_LOCK:
        METRICS[name] = METRICS.get(name, 0) + 1


def create_app(test_config=None):
    app = Flask(__name__)
    logger = setup_logging()

    app.config.from_mapping(
        APP_ENV=os.getenv("APP_ENV", "development").lower(),
        HOST=os.getenv("HOST", "0.0.0.0"),
        PORT=int(os.getenv("PORT", "5000")),
        SECRET_KEY=os.getenv("SECRET_KEY"),
        DATA_PATH=os.getenv("DATA_PATH", DEFAULT_DATA_PATH),
        MODEL_PATH=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
        REQUEST_TIMEOUT_SECONDS=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "5")),
        REDIS_URL=os.getenv("REDIS_URL", ""),
        RATE_LIMIT=int(os.getenv("RATE_LIMIT", "60")),
        RATE_WINDOW=int(os.getenv("RATE_WINDOW", "60")),
        AWS_SECRETS_ENABLED=os.getenv("AWS_SECRETS_ENABLED", "false"),
        AWS_SECRET_NAME=os.getenv("AWS_SECRET_NAME", ""),
        AWS_REGION=os.getenv("AWS_REGION", ""),
        SENTRY_DSN=os.getenv("SENTRY_DSN", ""),
    )

    if test_config:
        app.config.update(test_config)

    app.config["SECRET_KEY"] = load_secret_key(app.config)

    if app.config["APP_ENV"] == "production" and not app.config["SECRET_KEY"]:
        raise RuntimeError("SECRET_KEY must be set in production")

    app.secret_key = app.config.get("SECRET_KEY") or "test-only-secret"
    app.config["REDIS_CLIENT"] = init_redis_client(app.config["REDIS_URL"], logger)

    if app.config["SENTRY_DSN"]:
        sentry_sdk.init(dsn=app.config["SENTRY_DSN"], traces_sample_rate=0.1)
        logger.info("Sentry integration enabled")

    data = load_dataset(app.config["DATA_PATH"])
    model, model_load_error = load_model(app.config["MODEL_PATH"], logger)
    app.config["MODEL"] = model
    app.config["MODEL_LOAD_ERROR"] = model_load_error

    # Load model columns (IMPORTANT)
    columns_path = os.path.join(BASE_DIR, "model_columns.pkl")
    if os.path.exists(columns_path):
        app.config["MODEL_COLUMNS"] = joblib.load(columns_path)
        logger.info("Model columns loaded successfully")
    else:
        app.config["MODEL_COLUMNS"] = None
        logger.warning("model_columns.pkl not found")

    logger.info(
        "Startup status: %s",
        json.dumps(
            {
                "env": app.config["APP_ENV"],
                "data_path": app.config["DATA_PATH"],
                "model_path": app.config["MODEL_PATH"],
                "model_ready": model is not None,
            }
        ),
    )

    @app.before_request
    def apply_runtime_policies():
        increment_metric("total_requests")
        g.request_start = time.time()
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        if request.path != "/predict":
            return None

        redis_client = app.config["REDIS_CLIENT"]
        if not redis_client:
            return None

        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
        bucket = int(time.time() / app.config["RATE_WINDOW"])
        key = f"rl:{client_ip}:{bucket}"
        try:
            current = redis_client.incr(key)
            if current == 1:
                redis_client.expire(key, app.config["RATE_WINDOW"])
            if current > app.config["RATE_LIMIT"]:
                logger.warning("Rate limit exceeded for IP %s", client_ip)
                increment_metric("error_responses")
                return api_error("Rate limit exceeded. Please try again later.", 429)
        except RedisError:
            logger.warning("Redis operation failed, bypassing rate limit", exc_info=True)
        return None

    @app.after_request
    def log_request(response):
        response.headers["X-Request-ID"] = g.get("request_id", "")
        elapsed_ms = round((time.time() - g.get("request_start", time.time())) * 1000, 2)
        if response.status_code >= 400:
            increment_metric("error_responses")
        logger.info(
            json.dumps(
                {
                    "request_id": g.get("request_id"),
                    "path": request.path,
                    "method": request.method,
                    "status": response.status_code,
                    "elapsed_ms": elapsed_ms,
                }
            )
        )
        return response

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/contact", methods=["GET", "POST"])
    def contact():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip()
            message = request.form.get("message", "").strip()

            if not all([name, email, message]):
                flash("Please fill all contact form fields.", "danger")
                return render_template("contact.html")

            contact_path = os.path.join(BASE_DIR, "contact_messages.csv")
            contact_df = pd.DataFrame([{"name": name, "email": email, "message": message}])
            file_exists = os.path.exists(contact_path)
            contact_df.to_csv(contact_path, mode="a", index=False, header=not file_exists)
            flash("Message sent successfully. Thank you for contacting us!", "success")

        return render_template("contact.html")

    @app.route("/dataInsights")
    def dataInsights():
        return render_template("dataInsights.html")

    @app.route("/api/house-price-insights", methods=["GET"])
    def house_price_insights():
        avg_price_by_region = (
            data.groupby("region_name")["price"].mean().sort_values(ascending=False).head(5)
        )
        avg_price_per_sqft_by_region = (
            data.groupby("region_name")["value_per_sqft"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )

        return api_success(
            {
                "regions": avg_price_by_region.index.tolist(),
                "avg_prices": avg_price_by_region.values.tolist(),
                "avg_price_per_sqft": avg_price_per_sqft_by_region.values.tolist(),
            }
        )

    @app.route("/health", methods=["GET"])
    def health():
        checks = {
            "data_file_exists": os.path.exists(app.config["DATA_PATH"]),
            "model_file_exists": os.path.exists(app.config["MODEL_PATH"]),
            "model_loaded": app.config["MODEL"] is not None,
        }
        is_healthy = checks["data_file_exists"] and checks["model_loaded"]
        if is_healthy:
            return jsonify({"status": "ok", "checks": checks}), 200
        return jsonify({"status": "error", "message": "Service degraded", "checks": checks}), 503

    @app.route("/prediction")
    def prediction():
        locations = sorted(data["locality_name"].unique())
        regions = sorted(data["region_name"].unique())
        house_types = sorted(data["house_type"].unique())
        return render_template(
            "prediction.html",
            locations=locations,
            regions=regions,
            house_types=house_types,
            model_ready=app.config["MODEL"] is not None,
            model_error=app.config["MODEL_LOAD_ERROR"],
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        increment_metric("prediction_requests")
        logger.info("Prediction request received")

        try:
            # 1. Check model
            if app.config["MODEL"] is None:
                return api_error(
                    "Prediction model is unavailable. Please add a valid RandomForestModel.pkl.",
                    503,
                )

            # 2. Get input
            location = request.form.get("location")
            region = request.form.get("region")
            house_type = request.form.get("house-type")
            area = request.form.get("area", type=float)
            total_rooms = request.form.get("total_rooms", type=int)
            total_beds = request.form.get("total_beds", type=int)
            age = request.form.get("age", type=int)

            if not all([location, region, house_type, area, total_rooms, total_beds, age]):
                return api_error("Invalid input. All fields are required.", 400)

            # 3. Create DataFrame
            input_data = pd.DataFrame(
                [[location, region, area, house_type, total_rooms, total_beds, age]],
                columns=[
                    "locality_name",
                    "region_name",
                    "area",
                    "house_type",
                    "total_rooms",
                    "total_beds",
                    "age",
                ],
            )

            # 4. SAME preprocessing as training
            # Drop columns
            for col in ["locality_name", "region_name"]:
                if col in input_data.columns:
                    input_data = input_data.drop(col, axis=1)

            # One-hot encoding
            input_data = pd.get_dummies(input_data)

            # Align columns
            model_columns = app.config.get("MODEL_COLUMNS")
            if model_columns is None:
                return api_error("Model columns not available", 500)

            input_data = input_data.reindex(columns=model_columns, fill_value=0)

            # 5. Predict
            prediction = app.config["MODEL"].predict(input_data)[0]

            return api_success({"predicted_price": round(float(prediction), 2)})

        except FutureTimeoutError:
            logger.error("Prediction request timed out")
            return api_error("Prediction request timed out.", 504)

        except Exception:
            logger.exception("Unhandled prediction error")
            return api_error("Internal server error during prediction.", 500)

    @app.route("/metrics", methods=["GET"])
    def metrics():
        with METRICS_LOCK:
            metrics_copy = dict(METRICS)
        lines = [
            "# HELP app_total_requests Total HTTP requests",
            "# TYPE app_total_requests counter",
            f"app_total_requests {metrics_copy['total_requests']}",
            "# HELP app_prediction_requests Total prediction requests",
            "# TYPE app_prediction_requests counter",
            f"app_prediction_requests {metrics_copy['prediction_requests']}",
            "# HELP app_error_responses Total responses with status >= 400",
            "# TYPE app_error_responses counter",
            f"app_error_responses {metrics_copy['error_responses']}",
        ]
        return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; version=0.0.4"}

    return app


try:
    app = create_app()
except Exception:  # pragma: no cover
    app = None
