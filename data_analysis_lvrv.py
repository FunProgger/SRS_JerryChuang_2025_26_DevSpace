import pandas as pd

# Load your Excel file
df = pd.read_csv(r'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_volume\lvrv_volumes_20260204.csv')

# 1. Define the headers you are interested in based on your image
# lvm = lv mass, rvm = rv mass
target_cols = ['lv_vol', 'lvm', 'rv_vol', 'rvm']

# 2. Group by 'name' (which is your Case1, Case2, etc.)
# and calculate min and max for all frames
summary = df.groupby('name')[target_cols].agg(['min', 'max'])

# 3. Clean up the headers so they look like 'lv_vol_min', 'lv_vol_max', etc.
summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]

# 4. Display or save the result
print(summary)
timestamp = pd.Timestamp.now().strftime('%Y%m-%d_%H-%M-%S')
summary.to_csv(rf'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output cardiac data\lvrv_summary_{timestamp}.csv')