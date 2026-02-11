import argparse
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

parser = argparse.ArgumentParser(description='Plot mass vs thickness (avg and deviation).')
parser.add_argument(
    '--skip-csv',
    default=None,
    help='Optional CSV with case IDs to exclude (e.g., Exclude_case.csv).'
)
args = parser.parse_args()

# Load your Excel file
df_mass = pd.read_csv(r'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_volume\lvrv_volumes_20260204.csv')
df_thickness = pd.read_csv(r'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_thickness\thickness_analysis_20260209_145827\summary_thickness.csv')

# 1. Define the headers you are interested in based on your image
# lvm = lv mass, rvm = rv mass
target_cols_mass = ['lvm', 'rvm']
target_cols_thickness = ['avg_thickness']

header_mapping = {
    'name': ['name', 'folder_name', 'exclude_name'],
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

skip_tag = ''
if args.skip_csv:
    if os.path.exists(args.skip_csv):
        skip_df = pd.read_csv(args.skip_csv, sep=None, engine='python')
        skip_df.columns = skip_df.columns.str.strip()
        skip_col = resolve_column(skip_df, header_mapping['name'], 'case id')
        skip_ids = set(skip_df[skip_col].astype(str))
        mass_ids = set(df_mass['case_id'].astype(str))
        thickness_ids = set(df_thickness['case_id'].astype(str))
        present_in_mass = sorted(skip_ids.intersection(mass_ids))
        present_in_thickness = sorted(skip_ids.intersection(thickness_ids))
        print(f"Skip list size: {len(skip_ids)}")
        print(f"Found in mass: {len(present_in_mass)}")
        print(f"Found in thickness: {len(present_in_thickness)}")
        if present_in_mass:
            print(f"Mass matches: {present_in_mass}")
        if present_in_thickness:
            print(f"Thickness matches: {present_in_thickness}")
        df_mass = df_mass[~df_mass['case_id'].astype(str).isin(skip_ids)]
        df_thickness = df_thickness[~df_thickness['case_id'].astype(str).isin(skip_ids)]
        skip_tag = '_skip_included'
    else:
        print(f"Skip CSV not found: {args.skip_csv}. Continuing without exclusions.")

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

# Deviation (std) per case across frames
mass_dev = (
    df_mass
    .groupby('case_id')[['lvm', 'rvm']]
    .std()
    .reset_index()
    .rename(columns={'lvm': 'lvm_dev', 'rvm': 'rvm_dev'})
)

thickness_dev = (
    df_thickness
    .groupby(['case_id', type_col])[avg_col]
    .std()
    .reset_index()
)

thickness_dev_wide = (
    thickness_dev
    .pivot(index='case_id', columns=type_col, values=avg_col)
    .reset_index()
    .rename(columns={
        'lv': 'lv_thickness_dev',
        'rv': 'rv_thickness_dev'
    })
)

dev_df = pd.merge(thickness_dev_wide, mass_dev, on='case_id', how='inner')

# NEW: average thickness + mass deviation for deviation plot
dev_avg_df = pd.merge(
    thickness_wide[['case_id', 'lv_avg_thickness', 'rv_avg_thickness']],
    mass_dev,
    on='case_id',
    how='inner'
)

def add_regression(ax, x, y, line_color, text_color, text_xy=(0.05, 0.95)):
    xy = pd.concat([x, y], axis=1).dropna()
    if xy.empty or len(xy) < 2:
        return None
    x_vals = xy.iloc[:, 0].to_numpy()
    y_vals = xy.iloc[:, 1].to_numpy()
    res = linregress(x_vals, y_vals)
    slope = res.slope
    intercept = res.intercept
    line_x = np.linspace(x_vals.min(), x_vals.max(), 100)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color=line_color, linewidth=2)

    # R^2
    r2 = res.rvalue ** 2

    ax.text(
        text_xy[0],
        text_xy[1],
        f'R\u00b2 = {r2:.3f}',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        color=text_color
    )
    return res.pvalue

fig, (ax_avg, ax_dev) = plt.subplots(1, 2, figsize=(12, 5))

lv_avg_scatter = ax_avg.scatter(merged_df['lv_avg_thickness'], merged_df['avg_lvm'], alpha=0.7, color='red', label='LV')
rv_avg_scatter = ax_avg.scatter(merged_df['rv_avg_thickness'], merged_df['avg_rvm'], alpha=0.7, color='blue', label='RV')
ax_avg.set_xlabel('Average Thickness (mm)')
ax_avg.set_ylabel('Average Mass (g)')
ax_avg.set_title('Avg Mass vs Avg Thickness')
lv_avg_p = add_regression(ax_avg, merged_df['lv_avg_thickness'], merged_df['avg_lvm'], 'darkred', 'darkred', (0.05, 0.95))
rv_avg_p = add_regression(ax_avg, merged_df['rv_avg_thickness'], merged_df['avg_rvm'], 'navy', 'navy', (0.05, 0.85))
if lv_avg_p is not None:
    lv_avg_scatter.set_label(f"LV (p={lv_avg_p:.3g})")
if rv_avg_p is not None:
    rv_avg_scatter.set_label(f"RV (p={rv_avg_p:.3g})")
ax_avg.legend()

lv_dev_scatter = ax_dev.scatter(dev_avg_df['lv_avg_thickness'], dev_avg_df['lvm_dev'], alpha=0.7, color='orange', label='LV')
rv_dev_scatter = ax_dev.scatter(dev_avg_df['rv_avg_thickness'], dev_avg_df['rvm_dev'], alpha=0.7, color='teal', label='RV')
ax_dev.set_xlabel('Average Thickness (mm)')
ax_dev.set_ylabel('Mass Deviation (g)')
ax_dev.set_title('Mass Deviation vs Avg Thickness')
lv_dev_p = add_regression(ax_dev, dev_avg_df['lv_avg_thickness'], dev_avg_df['lvm_dev'], 'darkorange', 'darkorange', (0.05, 0.95))
rv_dev_p = add_regression(ax_dev, dev_avg_df['rv_avg_thickness'], dev_avg_df['rvm_dev'], 'darkslategray', 'darkslategray', (0.05, 0.85))
if lv_dev_p is not None:
    lv_dev_scatter.set_label(f"LV (p={lv_dev_p:.3g})")
if rv_dev_p is not None:
    rv_dev_scatter.set_label(f"RV (p={rv_dev_p:.3g})")
ax_dev.legend()

timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
plt.tight_layout()
plt.savefig(rf'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output cardiac data\avg_thickness_vs_mass_{timestamp}{skip_tag}.png', dpi=300)
plt.show()
merged_df.to_csv(rf'C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output cardiac data\avg_thickness_vs_mass_{timestamp}{skip_tag}.csv', index=False)