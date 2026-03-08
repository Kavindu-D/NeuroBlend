#!/usr/bin/env python3
"""
Filter NACC demographics/diagnosis CSV to only include
the 200 patients that have been preprocessed.

Reads the large NACC CSV file, filters to matching patient IDs,
and saves a smaller CSV with only relevant columns.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────────
# CONFIG — adjust these paths as needed
# ─────────────────────────────────────────────────────────────────

# The large NACC CSV (rename your .crdownload file to .csv first)
SOURCE_CSV = r"C:\Users\kdesh\Downloads\investigator_nacc71.csv"

# Your preprocessed patient folders
PREPROCESSED_DIR = r"D:\NACC_Matched_Pairs"

# Output filtered CSV
OUTPUT_CSV = r"C:\Users\kdesh\OneDrive\Documents\Year 4\Implementation\500_patients.csv"


def main():
    print("=" * 70)
    print("FILTER NACC DEMOGRAPHICS TO 200 PATIENTS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ─────────────────────────────────────────────────────────
    # STEP 1: Get list of your 200 patient IDs
    # ─────────────────────────────────────────────────────────

    preprocessed_path = Path(PREPROCESSED_DIR)
    if not preprocessed_path.exists():
        print(f"ERROR: Preprocessed directory not found: {PREPROCESSED_DIR}")
        sys.exit(1)

    patient_ids = sorted([
        d.name for d in preprocessed_path.iterdir()
        if d.is_dir() and d.name.startswith("NACC")
    ])

    print(f"Found {len(patient_ids)} preprocessed patients")
    print(f"First 5: {patient_ids[:5]}")
    print(f"Last 5:  {patient_ids[-5:]}")
    print()

    # ─────────────────────────────────────────────────────────
    # STEP 2: Load the large CSV
    # ─────────────────────────────────────────────────────────

    if not os.path.exists(SOURCE_CSV):
        # Try to find any CSV in Downloads
        downloads = Path(r"C:\Users\kdesh\Downloads")
        csv_files = list(downloads.glob("*.csv")) + list(downloads.glob("*.crdownload"))
        print(f"Source CSV not found at: {SOURCE_CSV}")
        print(f"CSV/crdownload files in Downloads:")
        for f in csv_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")
        print()
        print("Please update SOURCE_CSV path in the script and re-run.")
        sys.exit(1)

    print(f"Loading CSV: {SOURCE_CSV}")
    print("(This may take a moment for a 370MB file...)")

    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(SOURCE_CSV, encoding=encoding, low_memory=False)
            print(f"  Loaded with encoding: {encoding}")
            break
        except (UnicodeDecodeError, Exception) as e:
            print(f"  Failed with {encoding}: {e}")
            continue
    else:
        print("ERROR: Could not read CSV with any encoding")
        sys.exit(1)

    print(f"  Total rows: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    print()

    # ───────���─────────────────────────────────────────────────
    # STEP 3: Find the patient ID column
    # ─────────────────────────────────────────────────────────

    print("Looking for patient ID column...")
    print(f"All columns ({len(df.columns)}):")

    # Print first 30 columns to help identify
    for i, col in enumerate(df.columns[:30]):
        sample = df[col].dropna().head(3).tolist()
        print(f"  [{i}] {col}: {sample}")

    # Common NACC ID column names
    id_candidates = ['NACCID', 'naccid', 'NACC_ID', 'nacc_id', 'NACCAVAIL',
                     'Subject', 'subject', 'ID', 'PatientID', 'SubjectID',
                     'SUBJECT_ID', 'Participant_ID']

    id_column = None

    # Try exact match first
    for candidate in id_candidates:
        if candidate in df.columns:
            id_column = candidate
            print(f"\nFound ID column: '{id_column}'")
            break

    # Try case-insensitive match
    if id_column is None:
        col_lower = {c.lower(): c for c in df.columns}
        for candidate in id_candidates:
            if candidate.lower() in col_lower:
                id_column = col_lower[candidate.lower()]
                print(f"\nFound ID column (case-insensitive): '{id_column}'")
                break

    # Try to find column containing 'NACC' values
    if id_column is None:
        print("\nSearching for column containing NACC IDs...")
        for col in df.columns:
            try:
                sample_vals = df[col].dropna().head(100).astype(str)
                nacc_matches = sample_vals.str.contains('NACC', case=False).sum()
                if nacc_matches > 10:
                    id_column = col
                    print(f"  Found: '{id_column}' ({nacc_matches} NACC matches)")
                    break
            except Exception:
                continue

    if id_column is None:
        print("\nERROR: Could not find patient ID column automatically.")
        print("Please check the column names above and update the script.")
        print("\nFirst 5 rows of the dataframe:")
        print(df.head().to_string())
        sys.exit(1)

    # Show sample values
    print(f"Sample values in '{id_column}':")
    print(f"  {df[id_column].dropna().head(10).tolist()}")
    print()

    # ─────────────────────────────────────────────────────────
    # STEP 4: Match and filter
    # ─────────────────────────────────────────────────────────

    # Standardize IDs for matching
    df['_match_id'] = df[id_column].astype(str).str.strip().str.upper()
    patient_ids_upper = set([pid.upper() for pid in patient_ids])

    # Filter to our 200 patients
    mask = df['_match_id'].isin(patient_ids_upper)
    df_filtered = df[mask].copy()
    df_filtered.drop(columns=['_match_id'], inplace=True)

    print(f"Matching results:")
    print(f"  Preprocessed patients: {len(patient_ids)}")
    print(f"  Rows matched in CSV:   {len(df_filtered)}")
    print(f"  Unique patients matched: {df_filtered[id_column].nunique()}")
    print()

    # Find which patients didn't match
    matched_ids = set(df_filtered[id_column].astype(str).str.strip().str.upper())
    unmatched = [pid for pid in patient_ids if pid.upper() not in matched_ids]
    if unmatched:
        print(f"  WARNING: {len(unmatched)} patients not found in CSV:")
        for pid in unmatched[:10]:
            print(f"    {pid}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")
    else:
        print("  All patients matched!")
    print()

    # ─────────────────────────────────────────────────────────
    # STEP 5: Select relevant columns
    # ─────────────────────────────────────────────────────────

    # Common NACC diagnosis/demographic column names
    # We'll keep the ID + any columns related to diagnosis, demographics, AD
    priority_keywords = [
        'NACCID', 'ID',
        'SEX', 'GENDER', 'AGE', 'BIRTHYR', 'BIRTHMO',
        'RACE', 'ETHNIC', 'EDUC', 'MARISTAT',
        'NACCUDSD',  # NACC UDS diagnosis
        'NACCALZD',  # Alzheimer's diagnosis
        'NACCALZP',  # AD primary/contributing
        'DEMENTED', 'DEMENTIA',
        'NORMCOG',   # Normal cognition
        'IMPNOMCI',  # Impaired not MCI
        'MCI',       # Mild cognitive impairment
        'NACCETPR',  # Etiology primary
        'NACCTMCI',  # Type of MCI
        'CDRGLOB', 'CDRSUM',  # CDR scores
        'MMSE', 'NACCMMSE',   # Mini-mental state exam
        'MOCA', 'NACCMOCA',   # Montreal cognitive assessment
        'DIAGNOSIS', 'DX',
        'VISIT', 'VISITNUM', 'VISITYR', 'VISITMO',
        'NACCVNUM',  # Visit number
        'APOE',      # APOE genotype
        'BRAAK',     # Braak stage
    ]

    # Find matching columns
    selected_cols = []
    for col in df_filtered.columns:
        col_upper = col.upper()
        for keyword in priority_keywords:
            if keyword in col_upper:
                selected_cols.append(col)
                break

    # Always include the ID column
    if id_column not in selected_cols:
        selected_cols.insert(0, id_column)

    # Remove duplicates while preserving order
    seen = set()
    selected_cols_unique = []
    for col in selected_cols:
        if col not in seen:
            seen.add(col)
            selected_cols_unique.append(col)
    selected_cols = selected_cols_unique

    print(f"Selected {len(selected_cols)} relevant columns:")
    for col in selected_cols:
        n_non_null = df_filtered[col].notna().sum()
        print(f"  {col}: {n_non_null}/{len(df_filtered)} non-null")
    print()

    # If we found very few columns, keep all of them
    if len(selected_cols) < 5:
        print("WARNING: Found very few matching columns. Keeping ALL columns.")
        print("You can manually remove unwanted columns from the output CSV.")
        df_output = df_filtered
    else:
        df_output = df_filtered[selected_cols]

    # ─────────────────────────────────────────────────────────
    # STEP 6: Handle multiple visits (keep most recent)
    # ─────────────────────────────────────────────────────────

    # NACC often has multiple visits per patient
    # For AD prediction, we typically want the most recent diagnosis
    n_before = len(df_output)
    unique_patients = df_output[id_column].nunique()

    if len(df_output) > unique_patients:
        print(f"Multiple visits detected: {len(df_output)} rows for "
              f"{unique_patients} patients")

        # Try to find visit number/date column to sort by
        visit_cols = [c for c in df_output.columns
                      if any(kw in c.upper()
                             for kw in ['VISITNUM', 'NACCVNUM', 'VISITYR', 'VISIT'])]

        if visit_cols:
            sort_col = visit_cols[0]
            print(f"  Sorting by '{sort_col}' and keeping most recent visit")
            df_output = df_output.sort_values(sort_col, ascending=False)

        df_latest = df_output.drop_duplicates(subset=[id_column], keep='first')
        print(f"  Kept {len(df_latest)} rows (latest visit per patient)")

        # Also save the full version with all visits
        all_visits_path = OUTPUT_CSV.replace('.csv', '_all_visits.csv')
        df_output.to_csv(all_visits_path, index=False)
        print(f"  Saved all visits to: {all_visits_path}")

        df_output = df_latest

    # ─────────────────────────────────────────────────────────
    # STEP 7: Save filtered CSV
    # ─────────────────────────────────────────────────────────

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_output.to_csv(OUTPUT_CSV, index=False)

    print()
    print(f"Saved filtered CSV: {OUTPUT_CSV}")
    print(f"  Rows: {len(df_output)}")
    print(f"  Columns: {len(df_output.columns)}")
    print(f"  File size: {os.path.getsize(OUTPUT_CSV) / 1024:.1f} KB")
    print()

    # ─────────────────────────────────────────────────────────
    # STEP 8: Show diagnosis summary
    # ─────────────────────────────────────────────────────────

    print("=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print()

    # Try common diagnosis columns
    dx_cols = [c for c in df_output.columns
               if any(kw in c.upper()
                      for kw in ['NACCUDSD', 'NACCALZD', 'DEMENTED',
                                 'NORMCOG', 'DIAGNOSIS', 'DX', 'CDR'])]

    if dx_cols:
        for col in dx_cols:
            print(f"  {col}:")
            counts = df_output[col].value_counts(dropna=False)
            for val, count in counts.items():
                pct = count / len(df_output) * 100
                print(f"    {val}: {count} ({pct:.1f}%)")
            print()
    else:
        print("  No standard diagnosis columns found.")
        print("  Check the CSV manually for diagnosis information.")
        print()
        print("  Column preview:")
        print(df_output.head(3).to_string())

    # ─────────────────────────────────────────────────────────
    # STEP 9: AD vs Normal summary for model training
    # ─────────────────────────────────────────────────────────

    # NACCUDSD: 1=Normal, 2=Impaired not MCI, 3=MCI, 4=Dementia
    # NACCALZD: 0=No AD, 1=AD
    ad_col = None
    for candidate in ['NACCALZD', 'NACCUDSD', 'DEMENTED']:
        matches = [c for c in df_output.columns if candidate in c.upper()]
        if matches:
            ad_col = matches[0]
            break

    if ad_col:
        print("=" * 70)
        print(f"AD STATUS (from '{ad_col}')")
        print("=" * 70)
        print()
        counts = df_output[ad_col].value_counts(dropna=False)
        for val, count in counts.items():
            pct = count / len(df_output) * 100
            label = ""
            if ad_col.upper() == 'NACCUDSD':
                labels = {1: 'Normal Cognition', 2: 'Impaired not MCI',
                          3: 'MCI', 4: 'Dementia'}
                label = f" ({labels.get(val, 'Unknown')})"
            elif ad_col.upper() == 'NACCALZD':
                labels = {0: 'No AD', 1: 'AD Present'}
                label = f" ({labels.get(val, 'Unknown')})"
            print(f"  {val}{label}: {count} ({pct:.1f}%)")
        print()

    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()