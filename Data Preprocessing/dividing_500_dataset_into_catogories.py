import pandas as pd
import os
import shutil

# === CONFIG (SET YOUR PATHS) ===
FULL_CSV = r'C:\Users\kdesh\OneDrive\Documents\Year 4\Implementation\500_patients.csv'      # Full NACC download
FILTERED_CSV = r'C:\Users\kdesh\OneDrive\Documents\Year 4\Implementation\filtered_200_patients.csv'        # Your 200 subset  
ORIGINAL_DATA_DIR = r'D:\\NACC_Matched_Pairs'        # Scan here (originals untouched)
NEW_ORGANIZED_DIR = r'D:\\Organized_New_Only'          # Output here

# === STEP 1: LOAD & CATEGORIZE FULL DATASET ===
print("🔄 Loading full dataset...")
df_full = pd.read_csv(FULL_CSV)
df_filtered = pd.read_csv(FILTERED_CSV)

# Latest visit per patient
df_full_latest = df_full.loc[df_full.groupby('NACCID')['NACCVNUM'].idxmax()]

# Assign diagnosis
df_full_latest['primary_dx'] = 'Other'
df_full_latest.loc[df_full_latest['NACCALZD'] == 1, 'primary_dx'] = 'AD'
df_full_latest.loc[(df_full_latest['primary_dx'] == 'Other') & 
                   df_full_latest['NACCTMCI'].isin([1,2,3,4,5]), 'primary_dx'] = 'MCI'
df_full_latest.loc[(df_full_latest['primary_dx'] == 'Other') & 
                   (df_full_latest['NORMCOG'] == 1), 'primary_dx'] = 'Normal'

print("Full dataset breakdown:")
print(df_full_latest['primary_dx'].value_counts())

# === STEP 2: EXCLUDE YOUR 200 PATIENTS ===
filtered_naccid = set(pd.read_csv(FILTERED_CSV)['NACCID'].unique())
new_naccid = set(df_full_latest['NACCID']) - filtered_naccid

print(f"\n📊 New patients to organize: {len(new_naccid):,}")
print("Sample new NACCID:", list(new_naccid)[:5])

# === STEP 3: CREATE NEW STRUCTURE ===
print("\n📁 Creating output folders...")
os.makedirs(NEW_ORGANIZED_DIR, exist_ok=True)
for dx in ['Normal', 'AD', 'MCI', 'Other']:
    os.makedirs(os.path.join(NEW_ORGANIZED_DIR, dx), exist_ok=True)

# NACCID → diagnosis mapping for new patients only
new_dx_map = df_full_latest[df_full_latest['NACCID'].isin(new_naccid)].set_index('NACCID')['primary_dx'].to_dict()

# === STEP 4: SCAN & COPY (ORIGINALS UNTOUCHED) ===
print("\n🔍 Scanning data directory...")
copied_count = {'Normal': 0, 'AD': 0, 'MCI': 0, 'Other': 0}
skipped_count = 0

for root, dirs, files in os.walk(ORIGINAL_DATA_DIR):
    for folder_name in dirs:
        if folder_name in new_dx_map:
            dx = new_dx_map[folder_name]
            src_path = os.path.join(root, folder_name)
            dst_path = os.path.join(NEW_ORGANIZED_DIR, dx, folder_name)
            
            if os.path.exists(dst_path):
                skipped_count += 1
                print(f"⏭️ Skip (exists): {folder_name}")
                continue
            
            # COPY (original untouched)
            shutil.copytree(src_path, dst_path)
            copied_count[dx] += 1
            print(f"✅ {copied_count[dx]} {dx}: {folder_name}")
        elif folder_name.startswith('NACC'):
            skipped_count += 1  # Known filtered patient

print(f"\n🎉 SUMMARY:")
print(f"Copied: {sum(copied_count.values())} new patients")
for dx, count in copied_count.items():
    print(f"  📂 {dx}: {count}")
print(f"Skipped (filtered/existing): {skipped_count}")
print(f"\n📍 New organized data: {NEW_ORGANIZED_DIR}")
