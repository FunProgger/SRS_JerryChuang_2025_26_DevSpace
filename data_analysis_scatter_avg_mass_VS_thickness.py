import pandas as pd 
import matplotlib.pyplot as plt

# Load your Excel file
df_mass = pd.read_csv(r'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_volume\lvrv_volumes_20260204.csv')
df_thickness = pd.read_csv(r'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_thickness\thickness_analysis_20260209_145827\summary_thickness.csv')

# 1. Define the headers you are interested in based on your image
# lvm = lv mass, rvm = rv mass
target_cols_mass = ['lvm', 'rvm']
target_cols_thickness = ['avg_thickness']

header_mapping = {
    'name': ['name', 'folder_name'],
}

def resolve_column(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing {label} column. Expected one of: {candidates}")

case_col_mass = resolve_column(df_mass, header_mapping['name'], 'case id')
case_col_thickness = resolve_column(df_thickness, header_mapping['name'], 'case id')

df_mass = df_mass.rename(columns={case_col_mass: 'case_id'})
df_thickness = df_thickness.rename(columns={case_col_thickness: 'case_id'})

# Resolve thickness type and avg-thickness columns
type_col = resolve_column(df_thickness, ['thickness_type', 'type'], 'thickness type')
avg_col = resolve_column(df_thickness, ['avg_thickness', 'avg_thick'], 'avg thickness')

# Average per case and type (lv/rv)
thickness_summary = (
    df_thickness
    .groupby(['case_id', type_col])[avg_col]
    .mean()
    .reset_index()
)

# Pivot to one row per case with lv/rv columns
thickness_wide = (
    thickness_summary
    .pivot(index='case_id', columns=type_col, values=avg_col)
    .reset_index()
    .rename(columns={
        'lv': 'lv_avg_thickness',
        'rv': 'rv_avg_thickness'
    })
)

# Average mass per case (remove frame-level rows)
mass_summary = (
    df_mass
    .groupby('case_id')[['lvm', 'rvm']]
    .mean()
    .reset_index()
    .rename(columns={'lvm': 'avg_lvm', 'rvm': 'avg_rvm'})
)

# Merge thickness and mass summaries
merged_df = pd.merge(thickness_wide, mass_summary, on='case_id', how='inner')

fig, (ax_lv, ax_rv) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax_lv.scatter(merged_df['lv_avg_thickness'], merged_df['avg_lvm'], alpha=0.7, color='red')
ax_lv.set_xlabel('LV Average Thickness')
ax_lv.set_ylabel('LV Average Mass')
ax_lv.set_title('LV')

ax_rv.scatter(merged_df['rv_avg_thickness'], merged_df['avg_rvm'], alpha=0.7, color='blue')
ax_rv.set_xlabel('RV Average Thickness')
ax_rv.set_ylabel('RV Average Mass')
ax_rv.set_title('RV')

# TODO: regression model and line of best fit

# TODO: mass deviation vs thickness deviation (normalise across frame) 

timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
plt.tight_layout()
plt.savefig(rf'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output cardiac data\avg_thickness_vs_mass_{timestamp}.png', dpi=300)
plt.show()
merged_df.to_csv(rf'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output cardiac data\avg_thickness_vs_mass_{timestamp}.csv', index=False)