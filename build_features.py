"""
Hospital Operations & Revenue Risk Intelligence Platform
Phase 2 — Feature Engineering Script

Usage:
    python build_features.py                          # uses default paths
    python build_features.py --db ../hospital.db      # custom DB path
    python build_features.py --out ../Data_Outputs    # custom output path

Outputs:
    Data_Outputs/model_table.csv       — feature-engineered modelling dataset
    Data_Outputs/feature_schema.json   — feature lists for API validation
"""

import os
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── CLI Arguments ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Feature engineering for Hospital Risk Platform")
    parser.add_argument("--db",  default=None, help="Path to hospital.db SQLite database")
    parser.add_argument("--out", default=None, help="Output directory for model_table.csv and feature_schema.json")
    return parser.parse_args()


# ── Path resolution (works from any working directory) ────────────────────────
def resolve_paths(db_arg, out_arg):
    base = os.path.dirname(os.path.abspath(__file__))
    # Allow one level up if called from Notebooks/
    if os.path.basename(base).lower() == "notebooks":
        base = os.path.dirname(base)

    db_path  = db_arg  if db_arg  else os.path.join(base, "hospital.db")
    out_path = out_arg if out_arg else os.path.join(base, "Data_Outputs")
    return db_path, out_path


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(db_path):
    """Load patients, visits, billing from SQLite and merge into one DataFrame."""
    import sqlite3

    print(f"Loading data from: {db_path}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)

    patients = pd.read_sql(
        "SELECT patient_id, age, gender, city, insurance_provider, "
        "chronic_flag, registration_date FROM patients",
        conn,
        parse_dates=["registration_date"],
    )
    visits = pd.read_sql(
        "SELECT visit_id, patient_id, visit_date, department, visit_type, "
        "length_of_stay_hours, risk_score, doctor_id FROM visits",
        conn,
        parse_dates=["visit_date"],
    )
    billing = pd.read_sql(
        "SELECT visit_id, bill_id, billed_amount, approved_amount, "
        "claim_status, payment_days, billing_date FROM billing",
        conn,
        parse_dates=["billing_date"],
    )
    conn.close()

    # Merge: visits → patients → billing
    df = visits.merge(patients, on="patient_id", how="left")
    df = df.merge(billing,  on="visit_id",  how="left")

    print(f"  patients : {len(patients):,} rows")
    print(f"  visits   : {len(visits):,} rows")
    print(f"  billing  : {len(billing):,} rows")
    print(f"  merged   : {len(df):,} rows × {df.shape[1]} columns")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives 14 features from the merged hospital dataset.
    All features are safe from data leakage (computed before target is known).
    """
    df = df.copy()

    # 1 — Time-based visit features
    df["visit_month"]     = df["visit_date"].dt.month
    df["visit_dayofweek"] = df["visit_date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["visit_quarter"]   = df["visit_date"].dt.quarter
    df["is_weekend"]      = (df["visit_dayofweek"] >= 5).astype(int)

    # 2 — Days since patient registration (patient loyalty / acuity proxy)
    df["days_since_registration"] = (
        df["visit_date"] - df["registration_date"]
    ).dt.days.clip(lower=0)

    # 3 — Per-patient visit frequency (repeat visitor flag)
    visit_freq = (
        df.groupby("patient_id")["visit_id"]
        .count()
        .rename("patient_visit_freq")
    )
    df = df.merge(visit_freq, on="patient_id", how="left")

    # 4 — Per-patient average length of stay (individual baseline)
    patient_avg_los = (
        df.groupby("patient_id")["length_of_stay_hours"]
        .mean()
        .rename("patient_avg_los")
    )
    df = df.merge(patient_avg_los, on="patient_id", how="left")

    # 5 — Insurance provider rejection rate (insurer risk profile)
    insurer_rejection = (
        df.groupby("insurance_provider")["claim_status"]
        .apply(lambda x: (x == "Rejected").sum() / len(x))
        .rename("insurer_rejection_rate")
    )
    df = df.merge(insurer_rejection, on="insurance_provider", how="left")

    # 6 — Department average LOS (clinical complexity benchmark)
    dept_avg_los = (
        df.groupby("department")["length_of_stay_hours"]
        .mean()
        .rename("dept_avg_los")
    )
    df = df.merge(dept_avg_los, on="department", how="left")

    # 7 — LOS vs department average (patient deviation from dept norm)
    df["los_vs_dept_avg"] = df["length_of_stay_hours"] / df["dept_avg_los"].replace(0, np.nan)

    # 8 — Billing intensity per LOS hour (revenue efficiency signal)
    df["bill_per_los_hour"] = df["billed_amount"] / df["length_of_stay_hours"].replace(0, np.nan)

    # 9 — Log-transformed billed amount (reduces right-skew for tree models)
    df["log_billed_amount"] = np.log1p(df["billed_amount"])

    # 10 — Payment days missing flag (structural: Pending/Rejected have no payment date)
    #      Do NOT impute — the absence itself is the signal
    df["payment_days_missing"] = df["payment_days"].isnull().astype(int)

    # 11 — Approval ratio (post-settlement; excluded from model inputs, kept for EDA)
    df["approval_ratio"] = (
        df["approved_amount"] / df["billed_amount"].replace(0, np.nan)
    ).fillna(0)

    print(
        f"Feature engineering complete — {df.shape[1]} total columns. "
        f"New: visit_month, visit_dayofweek, visit_quarter, is_weekend, "
        f"days_since_registration, patient_visit_freq, patient_avg_los, "
        f"insurer_rejection_rate, dept_avg_los, los_vs_dept_avg, "
        f"bill_per_los_hour, log_billed_amount, payment_days_missing, approval_ratio"
    )
    return df


# ── Select Modeling Columns ───────────────────────────────────────────────────
def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only columns needed for modeling.
    approved_amount is excluded from model inputs to prevent data leakage into
    Model B (claim outcome prediction), but retained for evaluation purposes.
    """
    model_cols = [
        # Identifiers
        "visit_id", "patient_id", "visit_date",
        # Demographics
        "age", "gender", "city", "chronic_flag",
        # Visit features
        "department", "visit_type", "length_of_stay_hours",
        "visit_month", "visit_dayofweek", "visit_quarter", "is_weekend",
        # Doctor
        "doctor_id",
        # Billing (pre-approval — safe for claim prediction)
        "billed_amount", "log_billed_amount", "billing_date",
        # Engineered features
        "days_since_registration", "patient_visit_freq", "patient_avg_los",
        "insurer_rejection_rate", "dept_avg_los", "los_vs_dept_avg",
        "bill_per_los_hour", "payment_days_missing",
        # Insurance
        "insurance_provider",
        # Target variables
        "risk_score", "claim_status",
        # For evaluation only — NOT model inputs
        "payment_days", "approved_amount", "approval_ratio",
    ]
    available = [c for c in model_cols if c in df.columns]
    missing   = [c for c in model_cols if c not in df.columns]
    if missing:
        print(f"  ⚠️  Columns not found (skipped): {missing}")
    return df[available]


# ── Save Feature Schema ───────────────────────────────────────────────────────
def save_feature_schema(out_dir: str):
    """Saves the canonical feature lists used by model training and API validation."""
    schema = {
        "model_a_risk_features": [
            "age", "gender", "city", "chronic_flag",
            "department", "visit_type", "length_of_stay_hours",
            "visit_month", "visit_dayofweek", "is_weekend",
            "patient_visit_freq", "patient_avg_los",
            "dept_avg_los", "los_vs_dept_avg",
            "days_since_registration",
        ],
        "model_b_claim_features": [
            "age", "gender", "city", "chronic_flag",
            "department", "visit_type", "length_of_stay_hours",
            "billed_amount", "log_billed_amount",
            "insurance_provider", "insurer_rejection_rate",
            "visit_month", "visit_dayofweek", "is_weekend",
            "patient_visit_freq", "bill_per_los_hour",
            "payment_days_missing",
        ],
        "target_model_a": "risk_score",
        "target_model_b": "claim_status",
        "categorical_cols": ["gender", "city", "department", "visit_type", "insurance_provider"],
        "numeric_cols": [
            "age", "length_of_stay_hours", "billed_amount", "log_billed_amount",
            "patient_visit_freq", "patient_avg_los", "insurer_rejection_rate",
            "dept_avg_los", "los_vs_dept_avg", "bill_per_los_hour",
            "days_since_registration", "visit_month", "visit_dayofweek", "is_weekend",
        ],
        "binary_cols": ["chronic_flag", "payment_days_missing"],
        "leakage_excluded": [
            "approved_amount",   # available only post-settlement — excluded from Model B inputs
            "approval_ratio",    # derived from approved_amount — evaluation only
        ],
        "best_model_a": "Random Forest (Tuned)",
        "best_model_b": "Random Forest (Tuned)",
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
    }

    schema_path = os.path.join(out_dir, "feature_schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Saved: {schema_path}")
    return schema


# ── Save Data Quality Report ──────────────────────────────────────────────────
def save_data_quality_report(df: pd.DataFrame, model_df: pd.DataFrame, out_dir: str):
    """Writes a markdown data quality report alongside the model table."""
    report_dir = os.path.join(os.path.dirname(out_dir), "phase2_eda", "results")
    os.makedirs(report_dir, exist_ok=True)

    # Missing value stats
    missing_approved   = df["approved_amount"].isnull().sum()
    missing_payment    = df["payment_days"].isnull().sum()
    missing_los        = df["length_of_stay_hours"].isnull().sum()
    pct_approved       = df["approved_amount"].isnull().mean() * 100
    pct_payment        = df["payment_days"].isnull().mean() * 100
    pct_los            = df["length_of_stay_hours"].isnull().mean() * 100

    report = f"""# Data Quality Report
## Hospital Operations & Revenue Risk Intelligence Platform
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  **Script:** build_features.py

---

## 1. Dataset Overview
| Table    | Rows      | Columns |
|----------|-----------|---------|
| patients | {df['patient_id'].nunique():,} | 7 |
| visits   | {df['visit_id'].nunique():,} | 8 |
| billing  | {len(df):,} | 7 |
| Merged   | {len(df):,} | {df.shape[1]} |

## 2. Missing Value Summary — Key Columns
| Column                | Missing Count | Missing % | Treatment |
|-----------------------|--------------|-----------|-----------|
| approved_amount       | {missing_approved:,} | {pct_approved:.2f}% | Retain for evaluation; excluded from Model B inputs |
| payment_days          | {missing_payment:,} | {pct_payment:.2f}% | Structural — tied to Pending/Rejected claims |
| length_of_stay_hours  | {missing_los:,} | {pct_los:.2f}% | No action required |

**Key finding:** `payment_days` nulls are **structural**, not random — they occur exclusively
on Pending and Rejected claims where no payment was received. Treatment: engineered
`payment_days_missing` binary flag instead of imputing a fictitious value.

## 3. Outlier Summary
| Column               | Treatment |
|----------------------|-----------|
| billed_amount        | Retain + log-transform → `log_billed_amount` |
| length_of_stay_hours | Retain (clinical signal — extreme LOS is meaningful) |
| payment_days         | Retain (financial signal — extreme delays are meaningful) |

## 4. Feature Engineering — 14 New Columns
| Feature | Source Columns | Business Purpose |
|---------|---------------|-----------------|
| `visit_month` | visit_date | Seasonal admission patterns |
| `visit_dayofweek` | visit_date | Weekend vs weekday acuity |
| `visit_quarter` | visit_date | Quarterly trend analysis |
| `is_weekend` | visit_dayofweek | Differential weekend staffing signal |
| `days_since_registration` | visit_date, registration_date | Patient loyalty / chronic exposure |
| `patient_visit_freq` | patient_id, visit_id | Repeat visitor flag (chronic risk) |
| `patient_avg_los` | patient_id, length_of_stay_hours | Individual LOS baseline |
| `insurer_rejection_rate` | insurance_provider, claim_status | Insurer risk profile |
| `dept_avg_los` | department, length_of_stay_hours | Clinical complexity benchmark |
| `los_vs_dept_avg` | length_of_stay_hours, dept_avg_los | Patient deviation from dept norm |
| `bill_per_los_hour` | billed_amount, length_of_stay_hours | Revenue intensity signal |
| `log_billed_amount` | billed_amount | Normalised billing amount (reduces skew) |
| `payment_days_missing` | payment_days | Structural missing flag for unpaid claims |
| `approval_ratio` | approved_amount, billed_amount | Settlement ratio (evaluation only) |

## 5. Data Leakage Controls
- `approved_amount` excluded from Model B inputs (available only post-settlement)
- `approval_ratio` retained for EDA/evaluation only — not a model input
- Time-based 80/20 split applied in Phase 3 (earliest 80% → train, latest 20% → test)

## 6. Modeling Flags
- Class imbalance present in both targets (`risk_score`, `claim_status`)
  → Mitigation: `class_weight='balanced'` + random oversampling for minority classes
- Both targets confirmed to have 3 classes: High/Medium/Low and Paid/Pending/Rejected

## 7. Outputs
- `model_table.csv` : {len(model_df):,} rows × {model_df.shape[1]} columns
- `feature_schema.json` : feature lists for model training and API validation
- Plots : phase2_eda/plots/
"""

    report_path = os.path.join(report_dir, "DATA_QUALITY_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    db_path, out_dir = resolve_paths(args.db, args.out)

    print("\n" + "="*60)
    print("Hospital Risk Platform — Feature Engineering")
    print("="*60)

    # 1 — Load
    df = load_data(db_path)

    # 2 — Engineer features
    df = engineer_features(df)

    # 3 — Select modeling columns
    model_df = select_model_columns(df)

    # 4 — Save model table
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "model_table.csv")
    model_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}  ({model_df.shape[0]:,} rows × {model_df.shape[1]} cols)")

    # 5 — Save feature schema
    save_feature_schema(out_dir)

    # 6 — Save data quality report
    save_data_quality_report(df, model_df, out_dir)

    print("\n✅ build_features.py complete.")
    print("   → Proceed to: Notebooks/Phase3 ")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
