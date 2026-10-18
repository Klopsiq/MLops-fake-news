FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

# Та же версия, что в pyproject.toml. PostgreSQL driver нужен серверу.
RUN uv pip install --system --no-cache "mlflow==3.16.1" "psycopg2-binary==2.9.11"
RUN useradd --create-home --uid 10002 mlflow \
    && mkdir /mlartifacts \
    && chown mlflow:mlflow /mlartifacts

USER mlflow
WORKDIR /home/mlflow
EXPOSE 5000
