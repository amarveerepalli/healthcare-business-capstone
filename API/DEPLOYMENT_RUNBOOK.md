# Hospital Risk Intelligence API — Deployment & Operations Runbook

## Overview
This runbook covers building, deploying, testing, and operating the Hospital Risk Intelligence API.
The API serves two ML models:
- **Model A** — Patient Visit Risk Classifier (`/predict/risk`)
- **Model B** — Insurance Claim Outcome Classifier (`/predict/claim`)

All source files (`main.py`, `Dockerfile`, `requirements.txt`) are located in the `API/` folder.

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Docker | 24.x or later |
| Python | 3.10+ |
| Trained models | `Data_Outputs/model_a_risk.joblib`, `Data_Outputs/model_b_claim.joblib` |
| Feature schema | `Data_Outputs/feature_schema.json` |

---

## 2. Build Docker Image

Run from the **project root** (the folder containing `API/`, `Data_Outputs/`, `Notebooks/`, etc.):

```bash
# From project root
docker build -t hospital-risk-api:1.0.0 API/
```

> ⚠️ Note: The `API/` at the end of the command is the **build context** — it points to the `API/` folder which contains `main.py`, `Dockerfile`, and `requirements.txt`.

---

## 3. Run the Container

```bash
docker run -d \
  --name hospital-risk-api \
  -p 8000:8000 \
  -v $(pwd)/Data_Outputs:/app/models \
  -v $(pwd)/API/logs:/app/logs \
  hospital-risk-api:1.0.0
```

| Flag | Purpose |
|---|---|
| `-p 8000:8000` | Map container port 8000 to host port 8000 |
| `-v .../Data_Outputs:/app/models` | Mount trained model files and feature schema |
| `-v .../API/logs:/app/logs` | Persist prediction audit logs to host |

**Override paths via environment variables (optional):**
```bash
docker run -d \
  --name hospital-risk-api \
  -p 8000:8000 \
  -e MODEL_BASE=/app/models \
  -e LOG_PATH=/app/logs/prediction_audit.log \
  -v $(pwd)/Data_Outputs:/app/models \
  -v $(pwd)/API/logs:/app/logs \
  hospital-risk-api:1.0.0
```

---

## 4. Run Locally (Without Docker)

```bash
# From project root
pip install -r API/requirements.txt

# Set model path (optional — defaults to ../Data_Outputs relative to main.py)
export MODEL_BASE=$(pwd)/Data_Outputs

cd API
uvicorn main:app --reload --port 8000
```

---

## 5. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "timestamp": "2026-04-18T10:00:00.000000",
  "models_loaded": {"model_a": true, "model_b": true}
}
```

---

## 6. API Endpoints

### GET /health
Returns service status and model version.

### POST /predict/risk
Predicts patient visit risk level (High / Medium / Low).

**Sample Request:**
```bash
curl -X POST http://localhost:8000/predict/risk \
  -H "Content-Type: application/json" \
  -d '{
    "age": 58,
    "gender": "M",
    "city": "Hyderabad",
    "chronic_flag": 1,
    "department": "Cardiology",
    "visit_type": "ER",
    "length_of_stay_hours": 24.5,
    "visit_month": 6,
    "visit_dayofweek": 2,
    "is_weekend": 0,
    "patient_visit_freq": 4,
    "patient_avg_los": 18.3,
    "dept_avg_los": 20.1,
    "los_vs_dept_avg": 1.22,
    "days_since_registration": 365
  }'
```

**Sample Response:**
```json
{
  "visit_risk": "High",
  "confidence": {"High": 0.72, "Medium": 0.18, "Low": 0.10},
  "model_version": "1.0.0",
  "prediction_id": "a3f1c2d4e5b6f7a8",
  "timestamp": "2026-04-18T10:30:00.000000"
}
```

### POST /predict/claim
Predicts insurance claim outcome (Paid / Pending / Rejected).

**Sample Request:**
```bash
curl -X POST http://localhost:8000/predict/claim \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "F",
    "city": "Chennai",
    "chronic_flag": 0,
    "department": "Orthopedics",
    "visit_type": "OPD",
    "length_of_stay_hours": 6.0,
    "billed_amount": 25000.0,
    "log_billed_amount": 10.13,
    "insurance_provider": "HealthPlus",
    "insurer_rejection_rate": 0.15,
    "visit_month": 3,
    "visit_dayofweek": 1,
    "is_weekend": 0,
    "patient_visit_freq": 2,
    "bill_per_los_hour": 4166.67,
    "payment_days_missing": 1
  }'
```

**Sample Response:**
```json
{
  "claim_outcome": "Rejected",
  "confidence": {"Paid": 0.21, "Pending": 0.12, "Rejected": 0.67},
  "revenue_risk_flag": true,
  "model_version": "1.0.0",
  "prediction_id": "b4e2d3f5a6c7e8b9",
  "timestamp": "2026-04-18T10:31:00.000000"
}
```

> `revenue_risk_flag = true` when predicted outcome is `Rejected` AND `billed_amount > ₹20,000`.
> These claims are automatically routed for mandatory Finance team review.

---

## 7. HTTP Error Codes

| Code | Meaning | Action |
|---|---|---|
| 200 | Success | — |
| 422 | Validation error — field constraint violated | Check request schema against Pydantic model |
| 500 | Internal server error | Check `API/logs/prediction_audit.log` |

---

## 8. Prediction Audit Log

All predictions are appended to `API/logs/prediction_audit.log` (or `/app/logs/` in Docker):

```
2026-04-18 10:30:00,123 | {"endpoint": "/predict/risk", "prediction_id": "a3f1c2d4", "model_version": "1.0.0", "timestamp": "2026-04-18T10:30:00", "input_hash": "5d41402abc4b2a76"}
2026-04-18 10:31:00,456 | {"endpoint": "/predict/claim", "prediction_id": "b4e2d3f5", "model_version": "1.0.0", "timestamp": "2026-04-18T10:31:00", "input_hash": "7215ee9c7d9dc229"}
```

Each log entry records: endpoint, unique prediction ID, model version, UTC timestamp, and an MD5 hash of the input (for auditability without storing PII).

---

## 9. Interactive API Docs

When the service is running, access auto-generated documentation at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:**       http://localhost:8000/redoc

---

## 10. Stopping & Removing the Container

```bash
docker stop hospital-risk-api
docker rm hospital-risk-api
```

---

## 11. Drift Monitoring

Run the monitoring script monthly (or after any major data pipeline change):

```bash
# From project root
python phase6_monitoring/monitor.py

# With new production data
python phase6_monitoring/monitor.py --new-data path/to/new_data.csv
```

See `phase6_monitoring/results/retrain_decision.json` for the automated retrain recommendation.

---

## 12. Retraining & Redeployment

1. Run drift check: `python phase6_monitoring/monitor.py`
2. If `retrain_required: true`, retrain models in `Notebooks/Phase3_02_risk_model.ipynb and Phase3_03_claim_model.ipynb`
3. New `.joblib` files saved to `Data_Outputs/` (overwrite `model_a_risk.joblib` and `model_b_claim.joblib`)
4. Run Phase 4 evaluation notebook to confirm recall metrics meet production gate:
   - Model A: High-Risk Recall ≥ 0.65
   - Model B: Rejected Recall ≥ 0.70
5. Rebuild Docker image with new version tag:
   ```bash
   docker build -t hospital-risk-api:1.1.0 API/
   ```
6. Update `MODEL_VERSION = '1.1.0'` in `API/main.py` before rebuilding
7. Stop old container, start new one:
   ```bash
   docker stop hospital-risk-api && docker rm hospital-risk-api
   docker run -d --name hospital-risk-api -p 8000:8000 \
     -v $(pwd)/Data_Outputs:/app/models \
     -v $(pwd)/API/logs:/app/logs \
     hospital-risk-api:1.1.0
   ```
8. Run shadow mode for **2 weeks** — compare new model predictions against old before full cut-over
