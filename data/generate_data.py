"""
generate_data.py
================
Synthetic data generator for the Hospital Operations & Revenue Risk
Intelligence Platform capstone project.

Usage:
    python generate_data.py            # generates patients.csv, visits.csv, billing.csv
    python generate_data.py --seed 99  # reproducible with custom seed

Output files are written to the same directory as this script.
"""

import argparse
import os
import numpy as np
import pandas as pd

def generate(seed: int = 42, n_patients: int = 5000, n_visits: int = 25000):
    rng = np.random.default_rng(seed)
    out = os.path.dirname(os.path.abspath(__file__))

    # ── Patients ──────────────────────────────────────────────────────────────
    cities    = ['Hyderabad','Mumbai','Chennai','Pune','Bangalore','Delhi','Kolkata']
    insurers  = ['HealthPlus','SecureLife','MediCareX']
    genders   = ['M','F']

    patients = pd.DataFrame({
        'patient_id':        range(1, n_patients + 1),
        'age':               rng.integers(18, 90, n_patients),
        'gender':            rng.choice(genders, n_patients),
        'city':              rng.choice(cities, n_patients),
        'insurance_provider':rng.choice(insurers, n_patients),
        'chronic_flag':      rng.integers(0, 2, n_patients),
        'registration_date': pd.to_datetime('2025-01-01') +
                             pd.to_timedelta(rng.integers(0, 365, n_patients), unit='D')
    })
    patients.to_csv(os.path.join(out, 'patients.csv'), index=False)
    print(f'  patients.csv → {len(patients):,} rows')

    # ── Visits ────────────────────────────────────────────────────────────────
    departments  = ['Cardiology','Orthopedics','Neurology','ICU','ER',
                    'General','Pediatrics','Oncology']
    visit_types  = ['OPD','ER','ICU']
    risk_levels  = ['Low','Medium','High']

    visits = pd.DataFrame({
        'visit_id':             range(1, n_visits + 1),
        'patient_id':           rng.choice(patients['patient_id'], n_visits),
        'visit_date':           pd.to_datetime('2025-01-01') +
                                pd.to_timedelta(rng.integers(0, 365, n_visits), unit='D'),
        'department':           rng.choice(departments, n_visits),
        'visit_type':           rng.choice(visit_types, n_visits),
        'length_of_stay_hours': np.round(rng.exponential(18, n_visits).clip(1, 72), 2),
        'risk_score':           rng.choice(risk_levels, n_visits, p=[0.50, 0.30, 0.20]),
        'doctor_id':            rng.integers(100, 200, n_visits),
    })
    visits.to_csv(os.path.join(out, 'visits.csv'), index=False)
    print(f'  visits.csv  → {len(visits):,} rows')

    # ── Billing ───────────────────────────────────────────────────────────────
    claim_statuses = ['Paid','Pending','Rejected']
    billed = np.round(rng.uniform(3000, 50000, n_visits), 2)
    status = rng.choice(claim_statuses, n_visits, p=[0.60, 0.25, 0.15])
    approved = np.where(status == 'Paid',   billed,
               np.where(status == 'Pending', billed * rng.uniform(0, 1, n_visits),
                        0.0))
    pay_days = np.where(status == 'Paid',
                        rng.integers(5, 30, n_visits).astype(float), np.nan)

    billing = pd.DataFrame({
        'bill_id':         range(1, n_visits + 1),
        'visit_id':        visits['visit_id'],
        'billed_amount':   billed,
        'approved_amount': np.round(approved, 2),
        'claim_status':    status,
        'payment_days':    pay_days,
        'billing_date':    visits['visit_date'] + pd.to_timedelta(
                               rng.integers(0, 30, n_visits), unit='D')
    })
    billing.to_csv(os.path.join(out, 'billing.csv'), index=False)
    print(f'  billing.csv → {len(billing):,} rows')
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patients', type=int, default=5000)
    parser.add_argument('--visits',   type=int, default=25000)
    args = parser.parse_args()
    print(f'Generating data (seed={args.seed}) ...')
    generate(args.seed, args.patients, args.visits)
