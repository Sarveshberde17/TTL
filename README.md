# Maharashtra House Price Predictor

Production-hardened Flask ML app for Maharashtra house price estimation.

## Runtime requirements

- Python 3.10+
- `Cleaned_data.csv` in project root
- `RandomForestModel.pkl` in project root

Startup validation (`start.py`) fails fast when required artifacts are missing/invalid:
- data file must exist
- model file must exist
- model file must be non-empty
- in production, `SECRET_KEY` must be set

## Environment variables

- `APP_ENV`: `development` or `production`
- `HOST`: server bind host (default `0.0.0.0`)
- `PORT`: server port (default `5000`)
- `SECRET_KEY`: required in production
- `REQUEST_TIMEOUT_SECONDS`: model inference timeout for `/predict`
- `REDIS_URL`: Redis connection string for distributed rate limiting
- `RATE_LIMIT`: max requests per IP in window
- `RATE_WINDOW`: rate window in seconds
- `SENTRY_DSN`: optional Sentry DSN for error monitoring
- `AWS_SECRETS_ENABLED`: `true` to load secrets from AWS Secrets Manager
- `AWS_SECRET_NAME`: AWS secret name or ARN
- `AWS_REGION`: AWS region for Secrets Manager

## Local setup

```bash
python -m venv .venv
# activate virtual environment
pip install -r requirements.txt
python start.py
```

App runs at `http://localhost:5000`.

## Testing and linting

```bash
flake8 app.py start.py tests --max-line-length=120
pytest -q
```

## API behavior

- `GET /health`:
  - `200` with `{ "status": "ok", ... }` when healthy
  - `503` with `{ "status": "error", "message": "...", ... }` when degraded
- `POST /predict`:
  - standardized errors: `{ "status": "error", "message": "..." }`
  - standardized success: `{ "status": "ok", "predicted_price": ... }`
- `GET /metrics`:
  - exposes app request/error counters in Prometheus text format

## Docker-only deployment

Build:

```bash
docker build -t house-price-predictor .
```

Run:

```bash
docker run --rm -p 5000:5000 \
  -e APP_ENV=production \
  -e SECRET_KEY="<strong-random-secret>" \
  -e REDIS_URL="redis://host.docker.internal:6379/0" \
  -e REQUEST_TIMEOUT_SECONDS=5 \
  -e RATE_LIMIT=60 \
  -e RATE_WINDOW=60 \
  house-price-predictor
```

## Docker Compose deployment

`docker-compose.yml` enforces `SECRET_KEY` presence.

```bash
SECRET_KEY="<strong-random-secret>" docker compose up --build
```

## Redis setup

- **Local Docker Redis**
  - `docker run -d --name local-redis -p 6379:6379 redis:7`
  - set `REDIS_URL=redis://localhost:6379/0`
- **Cloud Redis**
  - use provider connection URL and set as `REDIS_URL`
  - if Redis is unavailable, limiter is bypassed and warning is logged

## Sentry setup

1. Create a project in Sentry.
2. Copy DSN and set `SENTRY_DSN` env var.
3. Deploy. Unhandled errors are captured automatically.

## Cloud deployment notes

- **Render**:
  - Build command: `pip install -r requirements.txt`
  - Start command: `python start.py`
  - Add `render.yaml` from repo root
  - Set env vars in Render dashboard: `APP_ENV=production`, `SECRET_KEY`, `REDIS_URL`, `SENTRY_DSN`
- **Railway**:
  - Use Dockerfile or Python runtime
  - Start command: `python start.py`
  - Set env vars: `APP_ENV=production`, `SECRET_KEY`, `REDIS_URL`, `SENTRY_DSN`, `PORT`
- **AWS (ECS/App Runner/Beanstalk)**:
  - Use Docker image
  - Start from `aws-ecs-task-definition.json` template in repo
  - Use AWS Secrets Manager with:
    - `AWS_SECRETS_ENABLED=true`
    - `AWS_SECRET_NAME=<secret-name>`
    - `AWS_REGION=<region>`
  - Ensure `RandomForestModel.pkl` is packaged or mounted securely

## AWS ECS deployment outline

1. Build and push image to ECR.
2. Update `image` and IAM roles in `aws-ecs-task-definition.json`.
3. Register task definition.
4. Create ECS service with ALB and port `5000`.
5. Configure CloudWatch log group `/ecs/flask-ml-api`.
