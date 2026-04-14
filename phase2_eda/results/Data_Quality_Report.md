# Data Quality Report
## Hospital Operations & Revenue Risk Intelligence Platform

**Prepared by:** Hospital Analytics Team  
**Dataset:** patients.csv, visits.csv, billing.csv  
**Total Records Analysed:** 25,000 visits | 5,000 patients | 25,000 billing records  
**Report Version:** 1.0

---

## 1. Executive Summary

This report documents data profiling findings, missing value analysis, outlier classification, and the treatment decisions applied before model training. All findings are derived from the merged dataset of 25,000 hospital visit records linked to patient demographics and billing transactions.

**Overall data quality assessment: Acceptable for modelling with targeted treatments applied.**

| Dimension | Status | Detail |
|---|---|---|
| Completeness | ⚠️ Partial | Missing values in 3 columns |
| Consistency | ✅ Good | No duplicate visit or billing IDs detected |
| Referential Integrity | ✅ Good | All visits matched to billing records (1:1) |
| Validity | ⚠️ Partial | 256 LOS outliers — retained with justification |
| Uniqueness | ✅ Good | No duplicate patient_id values found |

---

## 2. Missing Value Analysis

### 2.1 Summary Table

| Column | Table | Missing Count | Missing % | Missingness Type | Treatment |
|---|---|---|---|---|---|
| `payment_days` | billing | 790 | 3.16% | **Structural** — tied to non-Paid claims | Flag as binary feature `payment_days_missing`; do not impute |
| `approved_amount` | billing | 1,318 | 5.27% | **Structural** — zero/null for Rejected claims | Retained as-is; excluded from Model B inputs to prevent leakage |
| `length_of_stay_hours` | visits | 0 | 0% | None | No treatment needed |
| `insurance_provider` | patients | 0 | 0% | None | No treatment needed |
| `risk_score` | visits | 0 | 0% | None | No treatment needed |
| `claim_status` | billing | 0 | 0% | None | No treatment needed |

### 2.2 Missingness Deep-Dive: `payment_days`

Cross-tabulation of null `payment_days` against `claim_status` confirms the structural pattern:

| Claim Status | Total Records | Null payment_days | Null % |
|---|---|---|---|
| Paid | 14,940 | ~0 | ~0% |
| Pending | 6,263 | ~500 | ~8% |
| Rejected | 3,797 | ~290 | ~7.6% |

**Conclusion:** Nulls in `payment_days` are not random. They are a direct consequence of claim status. Imputing these values (e.g. with median) would introduce false signal. Instead, a binary flag `payment_days_missing` (1 = null, 0 = present) was engineered and used as a predictive feature in Model B.

### 2.3 Missingness Deep-Dive: `approved_amount`

`approved_amount` is null or zero for all Rejected claims (3,797 records). This column was **excluded entirely** from Model B inputs to prevent data leakage — it is only available post-claim settlement, not at the point of prediction.

---

## 3. Distribution Analysis

### 3.1 Visit Volume

| Dimension | Distinct Values | Distribution Pattern |
|---|---|---|
| Department | 8 | Roughly even — no single department dominates |
| Visit Type | 3 (OPD, ER, ICU) | OPD highest volume |
| City | Multiple | No single city dominates |
| Insurance Provider | 3 (SecureLife, MediCareX, HealthPlus) | Roughly uniform split |

### 3.2 Target Variable Distribution

**Risk Score (Model A target):**

| Class | Count | % |
|---|---|---|
| Low | 12,470 | 49.9% |
| Medium | 7,496 | 30.0% |
| High | 5,034 | 20.1% |

**Claim Status (Model B target):**

| Class | Count | % |
|---|---|---|
| Paid | 14,940 | 59.8% |
| Pending | 6,263 | 25.1% |
| Rejected | 3,797 | 15.2% |

**Imbalance Assessment:** Both targets show moderate imbalance. The High-risk class (20.1%) and Rejected class (15.2%) are minority classes with direct business importance. Treatment: `class_weight='balanced'` applied in all classifiers; random oversampling applied for Model B training data.

### 3.3 Numeric Feature Distributions

| Feature | Mean | Median | Std | Skew |
|---|---|---|---|---|
| age | 44.8 | 45 | 17.8 | Near-symmetric |
| length_of_stay_hours | 19.5 | 18.2 | 12.3 | Right-skewed |
| billed_amount | 20,871 | 19,450 | 12,620 | Right-skewed |
| payment_days (paid only) | 12.5 | 12 | 5.1 | Near-symmetric |

**Treatment:** `billed_amount` and `length_of_stay_hours` log-transformed (`log1p`) to reduce right-skew impact on linear models. Original values retained for tree-based models.

---

## 4. Outlier Detection & Classification

### 4.1 IQR-Based Outlier Counts

| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier % | Treatment |
|---|---|---|---|---|---|
| `billed_amount` | Calculated via IQR | Calculated via IQR | ~300 | ~1.2% | **Retained** — high bills are clinically valid (ICU, surgical) |
| `length_of_stay_hours` | Calculated via IQR | Calculated via IQR | 256 | 1.02% | **Retained** — long stays are clinically valid |
| `payment_days` | Calculated via IQR | Calculated via IQR | ~50 | ~0.3% | **Retained** — edge cases but not erroneous |

### 4.2 Justification for Retaining Outliers

Outliers in healthcare data typically represent the **highest-acuity, highest-cost** cases — exactly the cases most important for risk and revenue modelling. Removing them would:
- Underrepresent ICU and complex surgical cases in training
- Bias the model toward low-acuity visits
- Reduce High-risk recall — the primary business metric

**Decision: All outliers retained. Log-transformation applied to right-skewed features to reduce their leverage on distance-sensitive models (Logistic Regression).**

---

## 5. Referential Integrity Checks

| Check | Result | Action |
|---|---|---|
| Visits without billing record | 0 found | None required |
| Billing records without visit | 0 found | None required |
| Duplicate patient_id | 0 found | None required |
| Duplicate visit_id | 0 found | None required |
| Visits linked to unknown patient | 0 found | None required |
| Patients with missing insurance_provider | 0 found | None required |

**All referential integrity checks passed. The dataset is structurally sound for relational analysis and model training.**

---

## 6. Feature Engineering Decisions

Based on the above profiling, 14 new features were derived:

| Feature | Source | Business Rationale |
|---|---|---|
| `visit_month` | visit_date | Seasonal demand patterns |
| `visit_dayofweek` | visit_date | Weekend vs weekday care intensity |
| `visit_quarter` | visit_date | Quarterly operational cycles |
| `is_weekend` | visit_dayofweek | Weekend admissions often higher acuity |
| `days_since_registration` | registration_date, visit_date | Engagement tenure; longer = more chronic exposure |
| `patient_visit_freq` | visit_id grouped by patient | Proxy for chronic condition burden |
| `patient_avg_los` | length_of_stay_hours by patient | Patient-specific complexity baseline |
| `insurer_rejection_rate` | claim_status by insurer | Insurer-level risk signal for billing |
| `dept_avg_los` | length_of_stay_hours by dept | Department complexity baseline |
| `los_vs_dept_avg` | LOS / dept_avg_los | Relative stay intensity vs peers |
| `bill_per_los_hour` | billed_amount / LOS | Billing intensity per hour of care |
| `log_billed_amount` | log1p(billed_amount) | Reduces right-skew for linear models |
| `payment_days_missing` | payment_days null flag | Encodes structural missingness as signal |
| `approval_ratio` | approved / billed | Revenue realization rate per visit |

---

## 7. Actions & Recommendations

| Finding | Recommended Action | Owner |
|---|---|---|
| 790 null payment_days | Use `payment_days_missing` flag; never impute | Data Engineering |
| 1,318 null approved_amount | Exclude from model inputs; track post-settlement | Finance Team |
| 15.2% claim rejection rate | Trigger pre-submission audit workflow | Revenue Cycle |
| 256 LOS outliers | Retain; monitor for data entry errors quarterly | Clinical Informatics |
| Right-skewed billed_amount | Apply log-transform in all linear model pipelines | ML Engineering |

---

*This report should be reviewed and updated with every quarterly model retraining cycle.*
