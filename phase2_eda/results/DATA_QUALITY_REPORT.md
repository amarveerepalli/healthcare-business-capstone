# Data Quality Report
## Hospital Operations & Revenue Risk Intelligence Platform
**Generated:** 2026-04-19 00:09

## 1. Dataset Overview
| Table    | Rows       | Columns |
|----------|-----------|---------|
| patients | 5,000 | 7 |
| visits   | 25,000 | 8 |
| billing  | 25,000 | 7 |
| Merged   | 25,000 | 37 |

## 2. Missing Value Summary — Key Columns
| Column                | Missing Count | Missing % |
|-----------------------|--------------|-----------|
| approved_amount       | 1,318 | 5.27% |
| payment_days          | 790 | 3.16% |
| length_of_stay_hours  | 0 | 0.00% |

**Finding:** payment_days nulls are structural (tied to Pending/Rejected claims) — not random.
Engineered payment_days_missing as binary feature instead of imputing.

## 3. Outlier Summary
| Column               | Mild Outliers | Extreme Outliers | Treatment         |
|----------------------|--------------|-----------------|-------------------|
| billed_amount        | 369 | 4 | Retain + log-transform |
| length_of_stay_hours | 256 | 0 | Retain (clinical signal) |
| payment_days         | 490 | 19 | Retain (financial signal) |

## 4. Feature Engineering
14 new features engineered: visit_month, visit_dayofweek, visit_quarter, is_weekend,
days_since_registration, patient_visit_freq, patient_avg_los, insurer_rejection_rate,
dept_avg_los, los_vs_dept_avg, bill_per_los_hour, log_billed_amount,
payment_days_missing, approval_ratio

## 5. Modeling Flags
- Class imbalance present in both targets → use class_weight=balanced
- No data leakage: approved_amount excluded from Model B
- Time-based 80/20 split required for evaluation

## 6. Outputs
- model_table.csv: 25,000 rows × 32 columns
- feature_schema.json: saved
- Plots: phase2_eda/plots/
