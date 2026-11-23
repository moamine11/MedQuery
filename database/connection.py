"""Open psycopg connections using the local Compose defaults."""

import os

import psycopg


def connect() -> psycopg.Connection:
    """Connect using PG* settings and fall back to the Compose password."""
    settings = {
        "host": os.environ.get("PGHOST", "127.0.0.1"),
        "port": os.environ.get("PGPORT") or os.environ.get("POSTGRES_HOST_PORT") or "5433",
        "dbname": os.environ.get("PGDATABASE", "medical_rag"),
        "user": os.environ.get("PGUSER", "medical_user"),
        "connect_timeout": os.environ.get("PGCONNECT_TIMEOUT", "5"),
    }
    if "PGPASSWORD" not in os.environ and "POSTGRES_PASSWORD" in os.environ:
        settings["password"] = os.environ["POSTGRES_PASSWORD"]

    return psycopg.connect(**settings)
