"""
Hospital Operations & Revenue Risk Intelligence Platform
Phase 5 — FastAPI Deployment
Endpoints: /health, /predict/risk, /predict/claim
"""

import os, json, logging, hashlib
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE             = os.getenv('MODEL_BASE', os.path.join(os.path.dirname(__file__), '..', 'Data_Outputs'))
LOG_PATH         = os.getenv('LOG_PATH',   os.path.join(os.path.dirname(__file__), 'logs', 'prediction_audit.log'))
SCHEMA_PATH      = os.path.join(BASE, 'feature_schema.json')
MODEL_A_PATH     = os.path.join(BASE, 'model_a_risk.joblib')
MODEL_B_PATH     = os.path.join(BASE, 'model_b_claim.joblib')

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ── Load models & schema ──────────────────────────────────────────────────────
print('Loading models...')
model_a = joblib.load(MODEL_A_PATH)
model_b = joblib.load(MODEL_B_PATH)
with open(SCHEMA_PATH) as f:
    schema = json.load(f)

FEATURES_A = schema['model_a_risk_features']
FEATURES_B = schema['model_b_claim_features']
MODEL_VERSION = '1.0.0'

print(f'✅ Models loaded | Version: {MODEL_VERSION}')

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title='Hospital Risk Intelligence API',
    description='Predicts patient visit risk and insurance claim outcomes.',
    version=MODEL_VERSION
)

# ── Request Schemas ───────────────────────────────────────────────────────────
class RiskRequest(BaseModel):
    age:                    int   = Field(..., ge=0, le=130)
    gender:                 str   = Field(..., pattern='^(M|F|Other)$')
    city:                   str
    chronic_flag:           int   = Field(..., ge=0, le=1)
    department:             str
    visit_type:             str
    length_of_stay_hours:   float = Field(..., ge=0)
    visit_month:            int   = Field(..., ge=1, le=12)
    visit_dayofweek:        int   = Field(..., ge=0, le=6)
    is_weekend:             int   = Field(..., ge=0, le=1)
    patient_visit_freq:     float
    patient_avg_los:        float
    dept_avg_los:           float
    los_vs_dept_avg:        float
    days_since_registration: float

class ClaimRequest(BaseModel):
    age:                    int   = Field(..., ge=0, le=130)
    gender:                 str   = Field(..., pattern='^(M|F|Other)$')
    city:                   str
    chronic_flag:           int   = Field(..., ge=0, le=1)
    department:             str
    visit_type:             str
    length_of_stay_hours:   float = Field(..., ge=0)
    billed_amount:          float = Field(..., ge=0)
    log_billed_amount:      float
    insurance_provider:     str
    insurer_rejection_rate: float = Field(..., ge=0, le=1)
    visit_month:            int   = Field(..., ge=1, le=12)
    visit_dayofweek:        int   = Field(..., ge=0, le=6)
    is_weekend:             int   = Field(..., ge=0, le=1)
    patient_visit_freq:     float
    bill_per_los_hour:      float
    payment_days_missing:   int   = Field(..., ge=0, le=1)

# ── Response Schemas ──────────────────────────────────────────────────────────
class RiskResponse(BaseModel):
    visit_risk:       str
    confidence:       dict
    model_version:    str
    prediction_id:    str
    timestamp:        str

class ClaimResponse(BaseModel):
    claim_outcome:    str
    confidence:       dict
    revenue_risk_flag: bool
    model_version:    str
    prediction_id:    str
    timestamp:        str

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_prediction_id(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

def log_prediction(endpoint: str, prediction_id: str, input_data: dict):
    entry = {
        'endpoint':       endpoint,
        'prediction_id':  prediction_id,
        'model_version':  MODEL_VERSION,
        'timestamp':      datetime.utcnow().isoformat(),
        'input_hash':     hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
    }
    logger.info(json.dumps(entry))

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get('/health')
def health_check():
    return {
        'status':        'healthy',
        'model_version': MODEL_VERSION,
        'timestamp':     datetime.utcnow().isoformat(),
        'models_loaded': {'model_a': True, 'model_b': True}
    }

@app.post('/predict/risk', response_model=RiskResponse)
def predict_risk(req: RiskRequest):
    try:
        input_data = req.model_dump()
        X = pd.DataFrame([{f: input_data.get(f, np.nan) for f in FEATURES_A}])
        pred  = model_a.predict(X)[0]
        proba = model_a.predict_proba(X)[0]
        pid   = make_prediction_id(input_data)
        log_prediction('/predict/risk', pid, input_data)
        return RiskResponse(
            visit_risk    = pred,
            confidence    = dict(zip(model_a.classes_, proba.round(4).tolist())),
            model_version = MODEL_VERSION,
            prediction_id = pid,
            timestamp     = datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/claim', response_model=ClaimResponse)
def predict_claim(req: ClaimRequest):
    try:
        input_data = req.model_dump()
        X = pd.DataFrame([{f: input_data.get(f, np.nan) for f in FEATURES_B}])
        pred  = model_b.predict(X)[0]
        proba = model_b.predict_proba(X)[0]
        pid   = make_prediction_id(input_data)
        revenue_risk = (pred == 'Rejected' and input_data.get('billed_amount', 0) > 20000)
        log_prediction('/predict/claim', pid, input_data)
        return ClaimResponse(
            claim_outcome     = pred,
            confidence        = dict(zip(model_b.classes_, proba.round(4).tolist())),
            revenue_risk_flag = revenue_risk,
            model_version     = MODEL_VERSION,
            prediction_id     = pid,
            timestamp         = datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
