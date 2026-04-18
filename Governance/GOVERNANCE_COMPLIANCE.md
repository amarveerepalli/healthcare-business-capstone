# Hospital Risk Intelligence Platform
# Governance & Compliance Document
# Version 1.0 | Phase 6

---

## 1. System Overview

| Item | Detail |
|---|---|
| System Name | Hospital Operations & Revenue Risk Intelligence Platform |
| Version | 1.0.0 |
| Owner | Hospital Analytics Team |
| Review Cycle | Quarterly |
| Last Updated | April 2026 |

---

## 2. Models in Production

### Model A — Visit Risk Classifier
| Property | Value |
|---|---|
| Target | `risk_score` (High / Medium / Low) |
| Algorithm | Random Forest (tuned) |
| Training Strategy | Time-based 80/20 split |
| Imbalance Handling | `class_weight=balanced` |
| Key Business Metric | Recall for High-Risk class — achieved **0.283** on synthetic data ⚠️ |
| Threshold Target | ≥ 0.65 (production gate); current model is below threshold — retrain with clinical features recommended |

### Model B — Insurance Claim Outcome Classifier
| Property | Value |
|---|---|
| Target | `claim_status` (Paid / Pending / Rejected) |
| Algorithm | Random Forest (tuned) |
| Training Strategy | Time-based 80/20 split |
| Imbalance Handling | Random oversampling + `class_weight=balanced` |
| Key Business Metric | Recall for Rejected class — achieved **0.764** on synthetic data ✅ |
| Threshold Target | ≥ 0.65 (production gate); current model meets threshold |

---

## 3. Data Governance

### 3.1 Data Sources
- `hospital.db` (SQLite) — single source of truth, built in Phase 1 from raw CSV inputs
  - `patients` table — anonymised patient demographics (5,000 records)
  - `visits` table — hospital visit records (25,000 records)
  - `billing` table — insurance billing and claim outcomes (25,000 records)
- Raw CSV files (`patients.csv`, `visits.csv`, `billing.csv`) — used only during Phase 1 initial load; all downstream analysis reads from `hospital.db` via SQL queries

### 3.2 Data Quality Standards
- No personally identifiable information (PII) in model inputs
- All data is synthetic/anonymised for this capstone
- Missing values handled via engineered flag features (e.g. `payment_days_missing`)

### 3.3 Data Leakage Controls
- `approved_amount` excluded from Model B inputs (available only post-settlement)
- `approval_ratio` excluded from training features; retained for evaluation only
- Time-based split enforced to prevent future data leaking into training

---

## 4. Model Limitations

### Model A
- Does not incorporate real-time clinical vitals or lab results
- Performance may degrade for departments with very low visit counts in training
- Assumes `risk_score` labels in historical data are accurate ground truth

### Model B
- Does not encode insurer-specific contract rules or policy changes
- Pending class performance is inherently uncertain (outcome not yet resolved)
- New insurers not seen during training handled via OHE unknown category

---

## 5. Assumptions

- Patient demographic distribution remains stable over the deployment window
- Billing patterns remain consistent across quarters
- Hospital departments and visit types remain stable
- `claim_status` is final at labeling time (no re-adjudication)

---

## 6. Monitoring & Drift Detection

### 6.1 PSI Thresholds
| PSI Range | Status | Action |
|---|---|---|
| < 0.10 | Stable | No action required |
| 0.10 – 0.20 | Watch | Investigate root cause |
| > 0.20 | Drift | Retrain within 30 days |

**Current Drift Status (Phase 6 Run — April 2026):**

| Feature | PSI | Status |
|---|---|---|
| `days_since_registration` | **1.2785** | 🔴 Drift — retrain within 30 days |
| `age` | 0.0021 | ✅ Stable |
| `length_of_stay_hours` | 0.0021 | ✅ Stable |
| `billed_amount` | 0.0030 | ✅ Stable |
| `patient_visit_freq` | 0.0004 | ✅ Stable |
| `patient_avg_los` | 0.0010 | ✅ Stable |
| `insurer_rejection_rate` | 0.0013 | ✅ Stable |

> **Note:** The `days_since_registration` drift (PSI = 1.27) is a known structural artifact of the time-based train/test split — patients in the test set naturally have larger registration gaps than those in training. In live deployment, this feature must be computed on a rolling window basis and monitored monthly.

### 6.2 Monitored Features
- `age`, `length_of_stay_hours`, `billed_amount`
- `patient_visit_freq`, `insurer_rejection_rate`, `los_vs_dept_avg`
- `bill_per_los_hour`, `days_since_registration`, `patient_avg_los`

> **Note:** `payment_days` is **excluded** from monitored features — it is structurally NULL for Pending/Rejected claims and is not used as a model input. Its missingness is by design, not a data quality issue.

### 6.3 Prediction Distribution Monitoring
- Monitor week-over-week % of High / Medium / Low risk predictions
- Alert if High-risk % shifts by > 10 percentage points vs baseline
- Monitor Rejected claim % for sudden spikes indicating insurer policy change

---

## 7. Retraining Strategy

| Trigger | Condition | Timeline |
|---|---|---|
| Scheduled | Quarterly | Retrain on rolling 12-month window |
| Feature drift | PSI > 0.20 on any key feature | Within 30 days — **currently triggered** (`days_since_registration` PSI = 1.27) |
| Performance drop | High-risk recall < 0.65 | Immediate — **Model A (0.283) is below this threshold; clinical feature enrichment required before production deployment** |
| Business event | New insurer / new department added | Incremental retrain |

### Retraining Process
1. Collect new labeled data (minimum 3 months)
2. Run Phase 2 → Phase 3 pipeline on updated data
3. Evaluate against Phase 4 business metrics
4. Shadow mode deployment for 2 weeks
5. Full promotion if metrics meet thresholds

---

## 8. Audit & Compliance

- All predictions logged with: timestamp, model version, prediction ID, input hash
- Audit log location: `phase5_api/logs/prediction_audit.log`
- Log retention: minimum 12 months
- Model artifacts versioned and stored in `phase3_models/` (`model_a_risk.joblib`, `model_b_claim.joblib`)
- Model card maintained in `phase4_evaluation/model_card.json`
- Governance document: `Governance/GOVERNANCE_COMPLIANCE.md` (this file)
- Drift summary: `phase6_monitoring/drift_summary.csv`

---

## 9. Incident Response

### 9.1 Incident Classification

| Severity | Definition | Response SLA |
|---|---|---|
| P1 — Critical | API down, data breach, or model producing systematically wrong predictions at scale | Immediate — 1 hour |
| P2 — High | Recall drops below threshold, PSI > 0.20, or fairness gap exceeds 15 percentage points | Within 24 hours |
| P3 — Medium | API latency spike, single erroneous prediction, drift entering Watch zone | Within 72 hours |

### 9.2 Incident Response Procedures

**P1 — API Down or Critical Model Failure**
1. On-call engineer alerted via monitoring dashboard within 5 minutes of downtime
2. Rollback to previous model version: `docker run hospital-risk-api:previous_version`
3. Switch prediction endpoints to rule-based fallback (high billed_amount → flag as Rejected risk)
4. Notify Clinical Lead and Finance Lead within 1 hour
5. Root cause analysis completed within 24 hours; post-mortem documented

**P2 — Model Performance Degradation**
1. Freeze current model version — do not promote any new version
2. Investigate data pipeline for upstream changes (new insurer, new department, schema change)
3. Trigger emergency retraining on latest 3-month window
4. Shadow deploy new model for minimum 5 business days before promotion
5. Validate against Phase 4 business metric thresholds before live cutover

**P2 — Erroneous Prediction Reported by Clinical/Finance Team**
1. Log incident with: timestamp, prediction_id, input hash, reported output, expected output
2. Trace via audit log (`API/logs/prediction_audit.log`) to retrieve full input record
3. Check if input violated validation rules (`phase6_monitoring/results/validation_rules.json`)
4. If systemic (> 5 similar cases): escalate to P1
5. If isolated: document in model card limitations; retrain at next scheduled cycle

**P3 — Drift Watch Zone or Latency Spike**
1. Log PSI values and flag in monthly governance report
2. Schedule root cause investigation within 72 hours
3. If drift source identified (e.g. new patient registration patterns): retrain within 30 days
4. If latency spike: check Docker resource limits; scale container or optimise preprocessing

### 9.3 Escalation Path

| Stage | Contact | Method |
|---|---|---|
| First response | On-call Data Scientist | Monitoring alert / dashboard |
| Clinical impact | Hospital Operations Lead | Email + phone |
| Financial impact | Revenue Cycle Finance Lead | Email + phone |
| Compliance breach | Analytics Lead | Formal incident report |

### 9.4 Rollback Procedure

```bash
# Stop current container
docker stop hospital-risk-api

# Start previous version (always tag versions before deployment)
docker run -d --name hospital-risk-api -p 8000:8000 \
  -v $(pwd)/Data_Outputs:/app/models \
  hospital-risk-api:previous_version

# Verify health
curl http://localhost:8000/health
```

> All model versions must be tagged and retained for minimum 12 months to enable rollback at any point.

---

## 10. Sign-off

| Role | Name | Responsibility | Date |
|---|---|---|---|
| Data Scientist | Amarendranadh Veerepalli | Model development, evaluation, retraining | April 2026 |
| Analytics Lead | Amarendranadh Veerepalli | Drift monitoring, governance review | April 2026 |
| Clinical Stakeholder | *(Pending — Hospital Operations Team)* | Validate risk score label accuracy | — |
| Finance Lead | *(Pending — Revenue Cycle Team)* | Validate claim outcome label accuracy | — |

> **Document version:** 1.0 | Next review due: July 2026
