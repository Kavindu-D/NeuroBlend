# import pandas as pd

# # Load the CSV file
# df = pd.read_csv('500_patients.csv')

# # Get unique patients (NACCID) and their latest visit (max NACCVNUM)
# df_latest = df.loc[df.groupby('NACCID')['NACCVNUM'].idxmax()]

# # Normal: NORMCOG == 1
# normal_count = (df_latest['NORMCOG'] == 1).sum()

# # MCI: NACCTMCI != 8 and != -4 (non-missing MCI subtypes, e.g., 1-5)
# mci_count = ((df_latest['NACCTMCI'] != 8) & (df_latest['NACCTMCI'] != -4)).sum()

# # AD: NACCALZD == 1 (or use DEMENTED == 1 if preferred)
# ad_count = (df_latest['NACCALZD'] == 1).sum()

# print(f"Normal patients: {normal_count}")
# print(f"MCI patients: {mci_count}")
# print(f"Alzheimer's Disease patients: {ad_count}")
# print(f"Total unique patients: {len(df_latest)}")



import pandas as pd

df = pd.read_csv('500_patients.csv')
df_latest = df.loc[df.groupby('NACCID')['NACCVNUM'].idxmax()]

# Prioritize: AD first, then MCI, then Normal
df_latest['primary_dx'] = 'Other'
df_latest.loc[df_latest['NACCALZD'] == 1, 'primary_dx'] = 'AD'
df_latest.loc[(df_latest['primary_dx'] == 'Other') & (df_latest['NACCTMCI'] != 8) & (df_latest['NACCTMCI'] != -4), 'primary_dx'] = 'MCI'
df_latest.loc[(df_latest['primary_dx'] == 'Other') & (df_latest['NORMCOG'] == 1), 'primary_dx'] = 'Normal'

print(df_latest['primary_dx'].value_counts())
