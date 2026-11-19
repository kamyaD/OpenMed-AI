# OpenMed AI: Predicting and Preventing Medicine Stockouts Through Open AI: Backend

An AI-powered decision-support system that predicts medicine stockouts across a network of health facilities and helps supply-chain officials act before shortages occur.

Built for a health ministry managing essential medicines across ~500 facilities, where some hospitals run out of stock while others hold excess, deliveries arrive late, medicines expire unused, and inventory data is scattered across systems and spreadsheets.

> **Status:** backend complete and tested against synthetic data. The React frontend consumes this API. Model metrics come from simulated data and must not be read as real-world performance.

## What it does

- **Ingests** facility inventory, consumption and order/delivery data (CSV or JSON), normalises units and cadence, and keeps the raw source row for every rejected record.
- **Validates data quality**: missing fields, negative stock, invalid dates, duplicates, unknown facilities or medicines, zero-stock vs dispensing contradictions and inconsistent lead times. Quality reports by facility, medicine or source.
- **Predicts** the probability that a facility-medicine pair runs out of stock within the next 7 days, using leakage-safe features and chronological train/validation/test splits.
- **Categorises risk** into `STOCKED_OUT`, `CRITICAL`, `HIGH`, `MEDIUM` and `LOW` using an admin-editable, versioned policy.
- **Suggests redistribution** from low-risk donors with genuine surplus to at-risk facilities, respecting minimum stock, shelf life, distance and data freshness.
- **Requires human approval.** Suggestions are never executed automatically. Only authorised users can approve or reject, with a written rationale, and every decision is audit-logged.
- **Explains** predictions with deterministic facts plus model coefficient contributions, with an explicit disclaimer that these are associations, not causes.
- **Monitors** data freshness, completeness, prediction distribution, feature drift (PSI) and realised model performance, with a named owner and response for every alert.

## Responsible AI by design

- Models are never auto-promoted. Training creates candidates, and an administrator promotes one with a written rationale.
- Subgroup performance is evaluated by region, urban/rural, facility size, medicine category and data completeness. Flagged disparities must be explicitly acknowledged before promotion.
- Probabilities are labelled "calibrated" only if they pass a calibration check on held-out data.
- A model card documents intended use, limitations, escalation and prohibited autonomous actions.
- Role-based access control restricts facility users to their own facilities. An append-only audit log records sensitive actions.

## Tech stack

Django 5 · Django REST Framework · SimpleJWT · drf-spectacular (OpenAPI) · scikit-learn · pandas · PostgreSQL (SQLite for development) · Gunicorn · WhiteNoise · Docker · GitHub Actions

## Project structure

| App | Responsibility |
|---|---|
| `accounts` | Roles, permissions, JWT auth, audit log |
| `inventory` | Facilities, medicines, ingestion, data-quality validation and reporting |
| `ml` | Features, leakage checks, splits, training, calibration, model registry, prediction API |
| `decision` | Risk bands, coverage and replenishment gap, redistribution, approvals, dashboard endpoints, monitoring, rollback and retraining |

## Quick start

```bash
git clone <repo-url> && cd msp-backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # set DJANGO_SECRET_KEY

python manage.py migrate
python manage.py seed_demo_data --facilities 40 --medicines 15 --days 365
python manage.py create_demo_users
python manage.py check_leakage
python manage.py build_dataset
python manage.py train_models --dataset 1
python manage.py runserver
```

Then promote a model (admin), run `python manage.py run_daily`, and open **http://localhost:8000/api/docs/** for the interactive API documentation.

Demo users (development only): `admin1`, `analyst1`, `manager1`, `facility1`, all with the password printed by `create_demo_users`. Change it immediately outside local development.

## Key endpoints

| Area | Endpoint |
|---|---|
| Auth | `POST /api/v1/auth/token/` |
| Ingestion | `POST /api/v1/ingestion/{inventory,consumption,orders}/` |
| Data quality | `GET /api/v1/quality/report/`, `GET /api/v1/quality/issues/` |
| Models | `GET /api/v1/ml/models/`, `POST /api/v1/ml/models/{id}/promote/` |
| Predictions | `POST /api/v1/ml/predict/`, `POST /api/v1/ml/predict/batch/` |
| Dashboard | `GET /api/v1/dashboard/{overview,filters,detail,export}/` |
| Redistribution | `GET /api/v1/decision/redistribution/`, `POST .../{id}/approve/` |
| Monitoring | `POST /api/v1/monitoring/run/`, `GET /api/v1/monitoring/rules/` |
| Lifecycle | `GET /api/v1/lifecycle/status/`, `POST /api/v1/lifecycle/rollback/` |

## Daily operation

```bash
python manage.py run_daily    # batch predictions -> redistribution suggestions -> monitoring
```

Schedule it once a day after data ingestion. See `docs/RUNBOOK.md` for alert ownership, rollback and retraining procedures.

## Testing and CI

```bash
python manage.py test
ruff check .
```

GitHub Actions runs linting, migration checks, the test suite with coverage, and a Docker build on every push and pull request.

## Docker

```bash
cp .env.docker.example .env.docker     # set real secrets
GIT_COMMIT=$(git rev-parse --short HEAD) docker compose up --build
```

## Limitations

- Developed and evaluated on **synthetic data**. Real-world performance is unknown until a pilot.
- Stockout means end-of-day stock of zero; intra-day stockouts are not observed.
- Cost ratio (missed stockout vs false alarm) and risk band thresholds are **placeholders** that need stakeholder agreement.
- Pilot comparison facilities are not randomised, so outcome differences do not prove causation.
- UAT sign-off, user training and stakeholder-agreed parameters require people, not code. See `docs/UAT.md`.

## Documentation

- `docs/RUNBOOK.md`: operations, alert ownership, rollback and retraining
- `docs/UAT.md`: user acceptance test scenarios
- `/api/docs/`: OpenAPI (Swagger) documentation

## License

Choose a licence before publishing (for example MIT or Apache-2.0) and add a `LICENSE` file.
