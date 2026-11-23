# Medical RAG database

Run these commands from the repository root. Docker Compose uses the password in the local, Git-ignored `.env` file. Set `POSTGRES_PASSWORD` there if you are setting up a new workspace. For an existing database, use its current password.

## Start and migrate

```sh
docker compose up -d --wait
docker compose ps
cat database/migrations/[0-9]*.sql | docker compose exec -T postgres psql -X -U medical_user -d medical_rag -v ON_ERROR_STOP=1
```

Compose publishes PostgreSQL at `127.0.0.1:5433` by default. Set `POSTGRES_HOST_PORT` in `.env` to use a different host port. Migrations are versioned and safe to replay. See [database/SCHEMA.md](database/SCHEMA.md) for the schema and retrieval rules.

## Connect and test

Use Python 3.11 or later.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
set -a
. ./.env
set +a
.venv/bin/python -m unittest discover -s tests -v
```

Export the password directly if your Compose `.env` uses syntax that your shell cannot source. Python uses `PGPASSWORD` when set and otherwise accepts `POSTGRES_PASSWORD`. Override the connection defaults with `PGHOST`, `PGPORT`, `PGDATABASE`, and `PGUSER`. Without `PGPORT`, Python uses the exported `POSTGRES_HOST_PORT` or defaults to 5433. `PGCONNECT_TIMEOUT` defaults to five seconds.

Import `connect` from `database.connection`. Use `with connect() as conn` to commit on success or roll back on error and close the connection. Qualify application tables and views with `medical_rag.` in queries. For production retrieval, query the eligible views documented in the schema. The integration suite uses synthetic records with unique identifiers and rolls back each test transaction.
