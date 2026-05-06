import os
import sys

import joblib
from redis import Redis
from redis.exceptions import RedisError


class StartupValidationError(Exception):
    """Raised when required startup artifacts are not valid."""


def validate_startup(base_dir: str, app_env: str) -> tuple[str, str]:
    data_path = os.path.join(base_dir, "Cleaned_data.csv")
    model_path = os.path.join(base_dir, "RandomForestModel.pkl")

    if not os.path.exists(data_path):
        raise StartupValidationError(f"Required data file not found: {data_path}")

    if not os.path.exists(model_path):
        raise StartupValidationError(
            "Required model file not found: "
            f"{model_path}. Please place RandomForestModel.pkl in the project root."
        )

    if os.path.getsize(model_path) == 0:
        raise StartupValidationError(f"Model file is empty: {model_path}")

    try:
        joblib.load(model_path)
    except Exception as exc:
        raise StartupValidationError(f"Model load failed: {exc}") from exc

    if app_env == "production" and not os.getenv("SECRET_KEY"):
        aws_enabled = os.getenv("AWS_SECRETS_ENABLED", "false").lower() == "true"
        if not aws_enabled:
            raise StartupValidationError("SECRET_KEY must be set in production")

    if os.getenv("AWS_SECRETS_ENABLED", "false").lower() == "true":
        if not os.getenv("AWS_SECRET_NAME") or not os.getenv("AWS_REGION"):
            raise StartupValidationError(
                "AWS_SECRETS_ENABLED=true requires AWS_SECRET_NAME and AWS_REGION"
            )

    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            redis_client = Redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
        except RedisError:
            print("STARTUP CHECK WARNING: Redis unreachable, limiter will be disabled.", file=sys.stderr)

    return data_path, model_path


def fail(message: str) -> None:
    print(f"STARTUP CHECK FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_env = os.getenv("APP_ENV", "development").lower()
    try:
        data_path, model_path = validate_startup(base_dir, app_env)
    except StartupValidationError as exc:
        fail(str(exc))

    os.environ["DATA_PATH"] = data_path
    os.environ["MODEL_PATH"] = model_path

    # Import only after checks pass to avoid starting app in degraded state.
    from app import create_app  # pylint: disable=import-outside-toplevel
    from waitress import serve  # pylint: disable=import-outside-toplevel

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    app = create_app()

    if app_env == "development":
        app.run(host=host, port=port, debug=True)
    else:
        serve(app, host=host, port=port)


if __name__ == "__main__":
    main()
