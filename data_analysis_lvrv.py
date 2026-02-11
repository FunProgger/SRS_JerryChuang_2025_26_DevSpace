import pandas as pd
import argparse
import os

# Load your Excel file
df = pd.read_csv(r'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_volume\lvrv_volumes_20260204.csv')

parser = argparse.ArgumentParser(description='Summarize LV/RV volume and mass min/max.')
parser.add_argument(
    '--skip-csv',
    default=None,
    help='Optional CSV with case IDs to exclude (e.g., Exclude_case.csv).'
)

args = parser.parse_args()
header_mapping = {
    'name': ['name', 'folder_name', 'exclude_name'],
}

def resolve_column(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing {label} column. Expected one of: {candidates}")

case_col = resolve_column(df, header_mapping['name'], 'case id')
df = df.rename(columns={case_col: 'case_id'})

skip_tag = ''
if args.skip_csv:
    if os.path.exists(args.skip_csv):
        skip_df = pd.read_csv(args.skip_csv, sep=None, engine='python')
        skip_df.columns = skip_df.columns.str.strip()
        skip_col = resolve_column(skip_df, header_mapping['name'], 'case id')
        skip_ids = set(skip_df[skip_col].astype(str))
        df_ids = set(df['case_id'].astype(str))
        present_in_df = sorted(skip_ids.intersection(df_ids))
        print(f"Skip list size: {len(skip_ids)}")
        print(f"Found in data: {len(present_in_df)}")
        if present_in_df:
            print(f"Matches: {present_in_df}")
        df = df[~df['case_id'].astype(str).isin(skip_ids)]
        skip_tag = '_skip_included'
    else:
        print(f"Skip CSV not found: {args.skip_csv}. Continuing without exclusions.")

# 1. Define the headers you are interested in based on your image
# lvm = lv mass, rvm = rv mass
target_cols = ['lv_vol', 'lvm', 'rv_vol', 'rvm']

# 2. Group by 'case_id' (which is your Case1, Case2, etc.)
# and calculate min and max for all frames
summary = df.groupby('case_id')[target_cols].agg(['min', 'max'])

# 3. Clean up the headers so they look like 'lv_vol_min', 'lv_vol_max', etc.
summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]

# 4. Display or save the result
print(summary)
timestamp = pd.Timestamp.now().strftime('%Y%m-%d_%H-%M-%S')
summary.to_csv(rf'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output cardiac data\lvrv_summary_{timestamp}{skip_tag}.csv')