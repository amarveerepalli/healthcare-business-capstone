"""
Hospital Operations & Revenue Risk Intelligence Platform
Phase 6 — Model Monitoring & Drift Detection Script

Usage:
    python monitor.py                              # run drift check using model_table.csv (train vs test split)
    python monitor.py --new-data new_data.csv      # run drift check against new production data
    python monitor.py --report-only                # print existing drift summary without recomputing
    python monitor.py --out ./phase6_monitoring    # custom output directory

Outputs:
    phase6_monitoring/results/drift_detection_report.json
    phase6_monitoring/results/retrain_decision.json
    phase6_monitoring/results/validation_summary.json
    phase6_monitoring/plots/drift_chart.png
    Data_Outputs/drift_summary.csv
"""

import os
import json
import argparse
import warnings
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ── CLI Arguments ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Drift monitoring for Hospital Risk Platform")
    parser.add_argument("--new-data",    default=None,  help="Path to new production data CSV for drift comparison")
    parser.add_argument("--out",         default=None,  help="Output directory (default: ../phase6_monitoring)")
    parser.add_argument("--model-table", default=None,  help="Path to model_table.csv (default: ../Data_Outputs/model_table.csv)")
    parser.add_argument("--report-only", action="store_true", help="Print existing report without recomputing")
    return parser.parse_args()


# ── Path resolution ────────────────────────────────────────────────────────────
def resolve_paths(args):
    base = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(base).lower() == "notebooks":
        base = os.path.dirname(base)

    model_table = args.model_table or os.path.join(base, "Data_Outputs", "model_table.csv")
    out_dir     = args.out         or os.path.join(base, "phase6_monitoring")
    return base, model_table, out_dir


# ── Validation Rules ──────────────────────────────────────────────────────────
VALIDATION_RULES = {
    "input_validation_rules": {
        "age":                    {"min": 0,   "max": 120},
        "length_of_stay_hours":   {"min": 0,   "max": 200.0},
        "billed_amount":          {"min": 0,   "max": 200000.0},
        "chronic_flag":           {"allowed":  [0, 1]},
        "gender":                 {"allowed":  ["F", "M", "Other"]},
        "visit_type":             {"allowed":  ["ER", "ICU", "OPD"]},
        "visit_month":            {"min": 1,   "max": 12},
        "insurer_rejection_rate": {"min": 0.0, "max": 1.0},
    }
}


# ── Input Validation ──────────────────────────────────────────────────────────
def validate_record(record: dict, rules: dict = VALIDATION_RULES) -> list:
    """
    Validates a single input record against the defined rules.
    Returns a list of error strings. Empty list = valid record.

    Example:
        errors = validate_record({"age": 150, "gender": "M", ...})
        if errors:
            print("Validation failed:", errors)
    """
    errors = []
    for field, rule in rules["input_validation_rules"].items():
        val = record.get(field)
        if val is None:
            errors.append(f"{field}: missing")
            continue
        if rule.get("min") is not None and val < rule["min"]:
            errors.append(f"{field}: {val} < min {rule['min']}")
        if rule.get("max") is not None and val > rule["max"]:
            errors.append(f"{field}: {val} > max {rule['max']}")
        if rule.get("allowed") and val not in rule["allowed"]:
            errors.append(f"{field}: '{val}' not in allowed {rule['allowed']}")
    return errors


def validate_batch(df: pd.DataFrame, results_dir: str) -> dict:
    """Validates all records in a DataFrame and saves a summary."""
    sample_records = df.head(500).to_dict("records")
    total     = len(sample_records)
    valid     = 0
    invalid   = 0
    error_log = []

    for rec in sample_records:
        errs = validate_record(rec)
        if errs:
            invalid += 1
            error_log.append({"record_id": rec.get("visit_id"), "errors": errs})
        else:
            valid += 1

    summary = {
        "generated_at":   datetime.now().isoformat(),
        "total_checked":  total,
        "valid":          valid,
        "invalid":        invalid,
        "validation_rate": round(valid / total * 100, 2) if total > 0 else 0,
        "sample_errors":  error_log[:5],
    }

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Validation: {valid}/{total} records valid ({summary['validation_rate']}%)")
    return summary


# ── PSI Computation ───────────────────────────────────────────────────────────
def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.10  : Stable — no action required
    PSI 0.10–0.20: Watch  — investigate root cause
    PSI > 0.20  : Drift  — retrain within 30 days

    Args:
        expected : 1-D array from the training / reference distribution
        actual   : 1-D array from the test / production distribution
        bins     : number of percentile buckets (default 10)
    """
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    # Avoid log(0) — replace zeros with a small epsilon
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)

    return round(float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))), 4)


def drift_status(psi: float) -> str:
    if psi >= 0.20:
        return "YES"
    if psi >= 0.10:
        return "WATCH"
    return "NO"


# ── Run Drift Detection ───────────────────────────────────────────────────────
NUMERIC_DRIFT_FEATURES = [
    "age",
    "length_of_stay_hours",
    "billed_amount",
    "payment_days",
    "log_billed_amount",
    "patient_visit_freq",
    "patient_avg_los",
    "insurer_rejection_rate",
    "dept_avg_los",
    "los_vs_dept_avg",
    "bill_per_los_hour",
    "days_since_registration",
]


def run_drift_detection(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes PSI for each numeric feature comparing train vs test distributions.
    Returns a DataFrame with columns: feature, psi, drift_flag, train_mean, test_mean, mean_shift_pct
    """
    rows = []
    for feat in NUMERIC_DRIFT_FEATURES:
        if feat not in train_df.columns or feat not in test_df.columns:
            continue
        tr = train_df[feat].values.astype(float)
        te = test_df[feat].values.astype(float)

        psi         = compute_psi(tr, te)
        flag        = drift_status(psi)
        train_mean  = round(float(np.nanmean(tr)), 4)
        test_mean   = round(float(np.nanmean(te)), 4)
        shift_pct   = round((test_mean - train_mean) / abs(train_mean) * 100, 2) if train_mean != 0 else 0.0

        rows.append({
            "feature":        feat,
            "psi":            psi,
            "drift_flag":     flag,
            "train_mean":     train_mean,
            "test_mean":      test_mean,
            "mean_shift_pct": shift_pct,
        })

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


# ── Save Drift Chart ──────────────────────────────────────────────────────────
def save_drift_chart(drift_df: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)

    colors = [
        "#e74c3c" if f == "YES" else ("#f39c12" if f == "WATCH" else "#2ecc71")
        for f in drift_df["drift_flag"]
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(drift_df["feature"], drift_df["psi"], color=colors)
    ax.axvline(0.10, color="orange", linestyle="--", linewidth=1.5, label="Watch (0.10)")
    ax.axvline(0.20, color="red",    linestyle="--", linewidth=1.5, label="Retrain (0.20)")
    ax.set_title("Feature Drift — PSI Scores (Train vs Test / Production)")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.legend()

    # Annotate PSI values
    for i, (psi, flag) in enumerate(zip(drift_df["psi"], drift_df["drift_flag"])):
        ax.text(psi + 0.005, i, f"{psi:.4f}", va="center", fontsize=9,
                color="#c0392b" if flag == "YES" else "black")

    plt.tight_layout()
    chart_path = os.path.join(plots_dir, "drift_chart.png")
    plt.savefig(chart_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {chart_path}")


# ── Save Reports ──────────────────────────────────────────────────────────────
def save_drift_report(drift_df: pd.DataFrame, results_dir: str, data_outputs_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_outputs_dir, exist_ok=True)

    n_retrain = (drift_df["drift_flag"] == "YES").sum()
    n_watch   = (drift_df["drift_flag"] == "WATCH").sum()
    n_stable  = (drift_df["drift_flag"] == "NO").sum()

    recommended_action = "RETRAIN" if n_retrain > 0 else ("WATCH" if n_watch > 0 else "NO_ACTION")

    report = {
        "report_title":  "Feature Drift Detection Report",
        "generated_at":  datetime.now().isoformat(),
        "methodology":   "Population Stability Index (PSI) — Train vs Test split",
        "thresholds": {
            "stable":  "< 0.10",
            "watch":   "0.10 – 0.20",
            "retrain": "> 0.20",
        },
        "summary": {
            "total_features_checked": len(drift_df),
            "stable":  int(n_stable),
            "watch":   int(n_watch),
            "retrain": int(n_retrain),
        },
        "feature_results": drift_df.to_dict("records"),
        "interpretation": (
            "PSI < 0.10: Feature distribution is stable — no action required. "
            "PSI 0.10–0.20: Distribution shifting — investigate data pipeline. "
            "PSI > 0.20: Significant drift — schedule model retraining within 30 days."
        ),
        "recommended_action": recommended_action,
    }

    report_path = os.path.join(results_dir, "drift_detection_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    # Also save CSV summary to Data_Outputs/
    csv_path = os.path.join(data_outputs_dir, "drift_summary.csv")
    drift_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Retrain decision
    retrain_doc = {
        "generated_at":     datetime.now().isoformat(),
        "retrain_required": bool(n_retrain > 0),
        "reason":           f"{n_retrain} feature(s) with PSI > 0.20" if n_retrain > 0 else "All features stable",
        "drifted_features": drift_df[drift_df["drift_flag"] == "YES"][["feature", "psi", "mean_shift_pct"]].to_dict("records"),
        "recommended_action": recommended_action,
        "action_deadline":   "Within 30 days" if n_retrain > 0 else "N/A",
    }
    retrain_path = os.path.join(results_dir, "retrain_decision.json")
    with open(retrain_path, "w") as f:
        json.dump(retrain_doc, f, indent=2)
    print(f"  Saved: {retrain_path}")

    return report


# ── Print Console Summary ─────────────────────────────────────────────────────
def print_drift_summary(drift_df: pd.DataFrame):
    print("\n" + "="*60)
    print("DRIFT DETECTION SUMMARY")
    print("="*60)
    print(f"  Total features checked : {len(drift_df)}")
    print(f"  🟢 Stable  (PSI < 0.10) : {(drift_df['drift_flag'] == 'NO').sum()}")
    print(f"  🟡 Watch   (PSI 0.10–0.20): {(drift_df['drift_flag'] == 'WATCH').sum()}")
    print(f"  🔴 Retrain (PSI > 0.20) : {(drift_df['drift_flag'] == 'YES').sum()}")
    print()

    drifted = drift_df[drift_df["drift_flag"] == "YES"]
    if len(drifted) > 0:
        print("  🔴 FEATURES REQUIRING RETRAINING:")
        for _, row in drifted.iterrows():
            print(f"     {row['feature']:<35} PSI={row['psi']:.4f}  shift={row['mean_shift_pct']:+.1f}%")
        print("\n  ⚠️  ACTION: Retrain both models within 30 days.")
    else:
        print("  ✅ All features stable — no retraining required.")
    print("="*60 + "\n")


# ── Generate New Data Sample ──────────────────────────────────────────────────
def generate_new_data_sample(df: pd.DataFrame, out_dir: str, n: int = 100):
    """
    Generates a synthetic new data sample CSV by adding drift to days_since_registration.
    Used for demonstrating drift detection with unseen production data.
    """
    sample = df.sample(n=min(n, len(df)), random_state=99).copy()

    # Simulate production drift: days_since_registration grows over time
    sample["days_since_registration"] = (
        sample["days_since_registration"] + np.random.normal(loc=110, scale=20, size=len(sample))
    ).clip(lower=0)

    sample_path = os.path.join(out_dir, "new_data_sample.csv")
    sample.to_csv(sample_path, index=False)
    print(f"  Saved: {sample_path}  ({len(sample)} rows)")
    return sample


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args       = parse_args()
    base, model_table_path, out_dir = resolve_paths(args)

    results_dir     = os.path.join(out_dir, "results")
    plots_dir       = os.path.join(out_dir, "plots")
    data_outputs    = os.path.join(base, "Data_Outputs")

    print("\n" + "="*60)
    print("Hospital Risk Platform — Monitoring & Drift Detection")
    print("="*60)

    # Report-only mode
    if args.report_only:
        report_path = os.path.join(results_dir, "drift_detection_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            drift_df = pd.DataFrame(report["feature_results"])
            print_drift_summary(drift_df)
        else:
            print("  No existing report found. Run without --report-only to generate one.")
        return

    # Load model table
    if not os.path.exists(model_table_path):
        print(f"  ERROR: model_table.csv not found at {model_table_path}")
        print("  Run build_features.py first to generate it.")
        return

    df = pd.read_csv(model_table_path, parse_dates=["visit_date"])
    print(f"  Loaded model_table.csv: {df.shape[0]:,} rows")

    # Time-based 80/20 split → train = reference distribution
    df_sorted  = df.sort_values("visit_date").reset_index(drop=True)
    split_idx  = int(len(df_sorted) * 0.80)
    train_df   = df_sorted.iloc[:split_idx]
    test_df    = df_sorted.iloc[split_idx:]
    print(f"  Split: train={len(train_df):,} rows | test={len(test_df):,} rows")

    # If new data provided, use it as the "actual" distribution instead
    if args.new_data:
        if not os.path.exists(args.new_data):
            print(f"  ERROR: New data file not found: {args.new_data}")
            return
        new_df  = pd.read_csv(args.new_data)
        test_df = new_df
        print(f"  Using new data for drift comparison: {len(test_df):,} rows from {args.new_data}")
    else:
        # Generate a new_data_sample.csv for demo purposes if not already present
        sample_path = os.path.join(out_dir, "new_data_sample.csv")
        if not os.path.exists(sample_path):
            print("  Generating new_data_sample.csv for drift demonstration...")
            generate_new_data_sample(df_sorted, out_dir)

    # 1 — Input validation
    print("\n[1] Input Validation")
    validate_batch(df_sorted, results_dir)

    # 2 — Drift detection
    print("\n[2] Feature Drift Detection (PSI)")
    drift_df = run_drift_detection(train_df, test_df)

    # 3 — Save chart
    print("\n[3] Saving Drift Chart")
    save_drift_chart(drift_df, plots_dir)

    # 4 — Save reports
    print("\n[4] Saving Reports")
    save_drift_report(drift_df, results_dir, data_outputs)

    # 5 — Print summary
    print_drift_summary(drift_df)

    print("✅ monitor.py complete.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
