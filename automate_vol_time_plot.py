import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import argparse
import numpy as np
from datetime import datetime

r"""
Automated Volume-Time Plot Generation with Quality Assessment

This module generates volume versus time plots for cardiac left ventricle (LV) and 
right ventricle (RV) analysis from CSV data files. It processes volumetric data 
organized by patient and creates visualization plots with myocardial mass measurements.

The script performs the following operations:
- Recursively scans directories for CSV files containing cardiac volume data
- Filters out rows marked with asterisks in skip_lvm or skip_rvm columns
- Separates data by patient name and computes volume/mass statistics
- Generates dual-axis plots showing volume and myocardial mass with error bands
- Plots variation of points from the mean for LVM and RVM measurements
- Conducts quality assessment of measurements based on ±10% error tolerance
- Generates a separate cross-patient mass summary plot
- Saves plots and generates a quality assessment summary CSV

Parameters
----------
-mdir : str
    Path to the directory containing CSV files. Can be a parent directory with 
    subdirectories (multiple .csv files) or a single directory (single .csv file).
-o : str, optional
    Output directory for saving plots and quality assessment CSV. If not specified, 
    a timestamped folder will be created in the input directory.

CSV Input Format
----------------
Expected columns in input CSV files:
- frame : int
    Frame number
- lv_vol : float
    Left ventricle volume in mL
- lvm : float
    Left ventricle myocardial mass in g
- rv_vol : float
    Right ventricle volume in mL
- rvm : float
    Right ventricle myocardial mass in g
- name : str
    Patient identifier for separating multiple patients in one CSV
- skip_lvm : str, optional
    Marker column; rows containing '*' are excluded from LV analysis
- skip_rvm : str, optional
    Marker column; rows containing '*' are excluded from RV analysis

Output
------
Individual Patient Plots (PNG)
    Files named {patient}_vol_time_plot.png containing:
    - Left panel: LV volume (left y-axis), LV myocardial mass (right y-axis),
      and LVM deviation from mean (left y-axis, offset)
    - Right panel: RV volume (left y-axis), RV myocardial mass (right y-axis),
      and RVM deviation from mean (left y-axis, offset)
    - Error bands: ±10% deviation bands based on end-diastolic mass measurements
    - Connected deviation plots showing how each measurement deviates from the mean

Quality Assessment (CSV)
    File named quality_assessment_{timestamp}.csv with columns:
    - name : str
        Patient identifier
    - lvm : str
        Quality assessment ('good' or 'bad') for LV mass measurements
    - rvm : str
        Quality assessment ('good' or 'bad') for RV mass measurements
    - Summary statistics of assessment results

Mass Summary Plot (PNG)
    Separate image file (std_dev_tracking_{timestamp}.png) showing:
    - Top row: Average LVM and RVM for each patient with overall mean lines
    - Bottom row: Standard deviation from overall mean for each patient (LV/RV)

Patient Statistics (CSV)
    File named patient_statistics_{timestamp}.csv with columns:
    - patient : str
        Patient identifier
    - lvm_mean : float
        Mean LVM value for the patient
    - lvm_std : float
        Standard deviation of LVM measurements
    - rvm_mean : float
        Mean RVM value for the patient
    - rvm_std : float
        Standard deviation of RVM measurements

Notes
-----
- Error band tolerance is fixed at ±10% (ERROR_BAND = 0.10)
- Quality is marked 'bad' if any measurement falls outside the error band
- End-diastolic (ED) mass measurements (first frame) serve as reference values
- Deviation plots connect points with lines to show trends over time
- Plots are saved at 300 DPI resolution with tight layout

Examples
--------
Basic usage::

    python automate_vol_time_plot.py -mdir "C:/Users/jchu579/Documents/SRS 2025_26/dev_space/output_volume"

With custom output directory::

    python automate_vol_time_plot.py -mdir C:/Users/jchu579/Documents/SRS 2025_26/bivme-data -o C:/output/path
"""

# Error band percentage (as decimal)
ERROR_BAND = 0.10  # ±10%

# Parse command-line arguments

parser = argparse.ArgumentParser(description='Generate volume vs time plots from CSV files')
parser.add_argument('-mdir', required=True, help='Path to the directory containing CSV files. Can be a parent directory with subdirectories (multiple .csv files) or just a directory (single .csv file).')
parser.add_argument('-o', dest='output_dir', default=None, help='Output directory for saving plots. If not specified, a new timestamped folder will be created.')
args = parser.parse_args()

dir = args.mdir
output_dir = args.output_dir

# If output directory not specified, create a timestamped folder
if output_dir is None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    error_band_pct = int(ERROR_BAND * 100)
    output_dir = os.path.join(dir, f"lvrv_volume_plots_err{error_band_pct}pct_{timestamp}")
    print(f"No output directory specified. Creating: {output_dir}")
else:
    print(f"Using output directory: {output_dir}")

os.makedirs(output_dir, exist_ok=True)

if dir == None:
    print("Directory is None")

# List to store quality results
quality_results = []

# List to store patient statistics for std dev tracking plot
patient_stats = []

# List to store violated frame percentages
violated_frame_stats = []

# Get all CSV files from subdirectories
csv_files = []
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

# Warning if no CSV files found
if not csv_files:
    print(f"Warning: No CSV files found in {dir}, its subdirectories or the specified path is incorrect.")
    exit()
else:
    print(f"Found {len(csv_files)} CSV file(s)")
    for file in csv_files:
        print(f" - {file}")

for file_path in csv_files:

    print(f"\nProcessing file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        
        # Skip if 'frame' column doesn't exist
        if 'frame' not in df.columns:
            print(f"⊘ Skipping file: 'frame' column not found")
            continue
        
        # Convert frame column to integer
        df['frame'] = df['frame'].astype(int)
        
        # Filter out rows where 'skip_lvm' or 'skip_rvm' columns contain '*'
        initial_rows = len(df)
        skip_mask = pd.Series([False] * len(df))
        
        if 'skip_lvm' in df.columns:
            skip_mask |= df['skip_lvm'].astype(str).str.contains('\*', na=False)
        
        if 'skip_rvm' in df.columns:
            skip_mask |= df['skip_rvm'].astype(str).str.contains('\*', na=False)
        
        if skip_mask.any():
            df = df[~skip_mask].reset_index(drop=True)
            skipped_rows = initial_rows - len(df)
            print(f"  ⊘ Skipped {skipped_rows} row(s) with '*' in skip_lvm or skip_rvm columns")
        
        # Check if 'name' column exists for patient separation
        if 'name' in df.columns:
            # Separate by patient name
            patients = df['name'].unique()
            
            for patient in patients:
                patient_df = df[df['name'] == patient].reset_index(drop=True)
                
                # Calculate averages and determine quality
                # lvm_mean = patient_df['lvm'].mean()
                # lvm_std = lvm_mean * 0.05  # ±5% error band
                # rvm_mean = patient_df['rvm'].mean()
                # rvm_std = rvm_mean * 0.05  # ±5% error band

                lvm_mean = patient_df['lvm'].mean()
                lv_ed_mass = patient_df['lvm'][0] # ED mass is most reliable mass measurement
                lvm_std = lv_ed_mass * ERROR_BAND  # Error band, using ed_mass as reference
                # print(f"LV mass {patient}: lv_ed_mass={lv_ed_mass}, lvm_mean={lvm_mean}, lvm_std={lvm_std}")
                rvm_mean = patient_df['rvm'].mean()
                rv_ed_mass = patient_df['rvm'][0] # ED mass is most reliable
                rvm_std = rv_ed_mass * ERROR_BAND  # Error band
                # print(f"RV mass {patient}: rv_ed_mass={rv_ed_mass}, rvm_mean={rvm_mean}, rvm_std={rvm_std}")

                 
                # Check if any values fall outside error band
                lvm_out_of_range = (patient_df['lvm'] < lvm_mean - lvm_std) | (patient_df['lvm'] > lvm_mean + lvm_std)
                rvm_out_of_range = (patient_df['rvm'] < rvm_mean - rvm_std) | (patient_df['rvm'] > rvm_mean + rvm_std)
                
                lvm_quality = 'bad' if lvm_out_of_range.any() else 'good'
                rvm_quality = 'bad' if rvm_out_of_range.any() else 'good'
                
                # Store results
                quality_results.append({
                    'name': patient,
                    'lvm': lvm_quality,
                    'rvm': rvm_quality
                })
                
                # Store statistics for std dev tracking plot
                lvm_std_dev = patient_df['lvm'].std()
                rvm_std_dev = patient_df['rvm'].std()
                patient_stats.append({
                    'patient': patient,
                    'lvm_mean': lvm_mean,
                    'lvm_std': lvm_std_dev,
                    'rvm_mean': rvm_mean,
                    'rvm_std': rvm_std_dev
                })

                # Store violated frame percentage (LV and RV separately)
                total_frames = len(patient_df)
                lvm_violated_frames = lvm_out_of_range.sum()
                rvm_violated_frames = rvm_out_of_range.sum()
                lvm_violated_pct = (lvm_violated_frames / total_frames) * 100 if total_frames > 0 else 0
                rvm_violated_pct = (rvm_violated_frames / total_frames) * 100 if total_frames > 0 else 0
                violated_frame_stats.append({
                    'patient': patient,
                    'lvm_violated_pct': lvm_violated_pct,
                    'rvm_violated_pct': rvm_violated_pct
                })
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col')
                ax1, ax2 = axes[0, 0], axes[0, 1]
                ax1_dev, ax2_dev = axes[1, 0], axes[1, 1]

                # === LV Plot ===
                ax1.set_xlabel('Frames', fontsize=11)
                ax1.set_ylabel('lv_vol (mL)', fontsize=11, color='blue')
                line1 = ax1.plot(patient_df['frame'], patient_df['lv_vol'], label='lv_vol', color='blue', linewidth=2, marker='o', markersize=4)
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.grid(True, alpha=0.3)
                ax1.set_xticks(range(0, int(patient_df['frame'].max()) + 5, 5))
                
                # secondary axis for lvm
                ax3 = ax1.twinx()
                ax3.set_ylabel('lvm (g)', fontsize=11, color='green')
                line2 = ax3.plot(patient_df['frame'], patient_df['lvm'], label='lv_myo_mass', color='green', linewidth=2, marker='s', markersize=4)
                ax3.tick_params(axis='y', labelcolor='green')
                
                # Add average lvm line with error band
                # lvm_mean = patient_df['lvm'].mean()
                # lvm_std = lvm_mean * 0.05  # ±5% error band
                line_avg_lvm = ax3.axhline(y=lvm_mean, color='green', linestyle='--', linewidth=2, label='lvm_avg', alpha=0.7)
                line_err_lvm = ax3.fill_between(patient_df['frame'], lvm_mean - lvm_std, lvm_mean + lvm_std, 
                                color='green', alpha=0.15, label='lvm_error_band')

                # Plot exceedance beyond ±10% band (separate subplot below)
                lvm_deviation_mag = np.abs(patient_df['lvm'] - lvm_mean)
                rvm_deviation_mag = np.abs(patient_df['rvm'] - rvm_mean)
                lvm_exceed = np.maximum(0, lvm_deviation_mag - lvm_std)
                rvm_exceed = np.maximum(0, rvm_deviation_mag - rvm_std)
                deviation_max = max(lvm_exceed.max(), rvm_exceed.max()) * 1.1

                ax1_dev.set_xlabel('Frames', fontsize=11)
                ax1_dev.set_ylabel('LVM Exceedance Beyond ±10% (g)', fontsize=10, color='darkgreen')
                ax1_dev.plot(patient_df['frame'], lvm_exceed, color='darkgreen', linewidth=1.5, alpha=0.7, linestyle='--')
                ax1_dev.scatter(patient_df['frame'], lvm_exceed, color='darkgreen', s=50, alpha=0.6, 
                               marker='^', label='lvm_exceed', edgecolors='darkgreen', linewidth=1)
                ax1_dev.tick_params(axis='y', labelcolor='darkgreen')
                ax1_dev.grid(True, alpha=0.3)
                ax1_dev.set_ylim(0, deviation_max)
                
                # Combined legend
                lines = line1 + line2
                lines.append(line_avg_lvm)
                dev_line = ax1_dev.get_lines()[0]
                lines.append(dev_line)
                labels = [l.get_label() for l in lines[:-2]] + ['lvm_avg', 'lvm_error_band', 'lvm_exceed']
                ax1.legend(lines, labels, loc='upper left', fontsize=10)
                ax1.set_title(f'LV Volume over Frames for {patient}')

                # === RV Plot ===
                ax2.set_xlabel('Frames', fontsize=11)
                ax2.set_ylabel('rv_vol (mL)', fontsize=11, color='orange')
                line3 = ax2.plot(patient_df['frame'], patient_df['rv_vol'], label='rv_vol', color='orange', linewidth=2, marker='o', markersize=4)
                ax2.tick_params(axis='y', labelcolor='orange')
                ax2.grid(True, alpha=0.3)
                ax2.set_xticks(range(0, int(patient_df['frame'].max()) + 5, 5))
                
                # secondary axis for rvm
                ax4 = ax2.twinx()
                ax4.set_ylabel('rvm (g)', fontsize=11, color='red')
                line4 = ax4.plot(patient_df['frame'], patient_df['rvm'], label='rv_myo_mass', color='red', linewidth=2, marker='s', markersize=4)
                ax4.tick_params(axis='y', labelcolor='red')
                
                # Add average rvm line with error band
                # rvm_mean = patient_df['rvm'].mean()
                # rvm_std = rvm_mean * 0.05  # ±5% error band
                line_avg_rvm = ax4.axhline(y=rvm_mean, color='red', linestyle='--', linewidth=2, label='rvm_avg', alpha=0.7)
                line_err_rvm = ax4.fill_between(patient_df['frame'], rvm_mean - rvm_std, rvm_mean + rvm_std, 
                                color='red', alpha=0.15, label='rvm_error_band')

                # Plot exceedance beyond ±10% band (separate subplot below) - using pre-calculated deviation_max for consistency
                ax2_dev.set_xlabel('Frames', fontsize=11)
                ax2_dev.set_ylabel('RVM Exceedance Beyond ±10% (g)', fontsize=10, color='darkred')
                ax2_dev.plot(patient_df['frame'], rvm_exceed, color='darkred', linewidth=1.5, alpha=0.7, linestyle='--')
                ax2_dev.scatter(patient_df['frame'], rvm_exceed, color='darkred', s=50, alpha=0.6, 
                               marker='^', label='rvm_exceed', edgecolors='darkred', linewidth=1)
                ax2_dev.tick_params(axis='y', labelcolor='darkred')
                ax2_dev.grid(True, alpha=0.3)
                ax2_dev.set_ylim(0, deviation_max)
                
                # Combined legend
                lines = line3 + line4
                lines.append(line_avg_rvm)
                dev_line = ax2_dev.get_lines()[0]
                lines.append(dev_line)
                labels = [l.get_label() for l in lines[:-2]] + ['rvm_avg', 'rvm_error_band', 'rvm_exceed']
                ax2.legend(lines, labels, loc='upper left', fontsize=10)
                ax2.set_title(f'RV Volume over Frames for {patient}')
                
                plt.tight_layout()
                
                # Save plot in output directory
                output_path = os.path.join(output_dir, f"{patient}_vol_time_plot.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved plot to: {output_path}")
    except Exception as e:
        print(f"✗ Error processing {file_path}: {str(e)}")
        plt.close()

# Save quality results to CSV
if quality_results:
    quality_df = pd.DataFrame(quality_results)
    
    # Count bad assessments
    bad_lvm_count = (quality_df['lvm'] == 'bad').sum()
    bad_rvm_count = (quality_df['rvm'] == 'bad').sum()
    total_count = len(quality_df)
    good_lvm_count = total_count - bad_lvm_count
    good_rvm_count = total_count - bad_rvm_count
    
    # Add summary columns
    quality_df['bad_lvm_count'] = bad_lvm_count
    quality_df['good_lvm_count'] = good_lvm_count
    quality_df['bad_rvm_count'] = bad_rvm_count
    quality_df['good_rvm_count'] = good_rvm_count
    quality_df['total_count'] = total_count
    
    # Save in output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv_path = os.path.join(output_dir, f'quality_assessment_{timestamp}.csv')
    quality_df.to_csv(output_csv_path, index=False)
    print(f"\n✓ Quality assessment saved to: {output_csv_path}")
    
    # Print summary
    print(f"\nQuality Summary:")
    print(f"  Total patients: {total_count}")
    print(f"  LVM - Bad: {bad_lvm_count}, Good: {good_lvm_count}")
    print(f"  RVM - Bad: {bad_rvm_count}, Good: {good_rvm_count}")

# Generate cross-patient mass summary plot
if patient_stats:
    stats_df = pd.DataFrame(patient_stats)

    x_pos = np.arange(len(stats_df))
    overall_lvm_mean = stats_df['lvm_mean'].mean()
    overall_rvm_mean = stats_df['rvm_mean'].mean()

    # Compute std from overall mean for each patient
    lvm_std_from_overall = np.sqrt(stats_df['lvm_std'] ** 2 + (stats_df['lvm_mean'] - overall_lvm_mean) ** 2)
    rvm_std_from_overall = np.sqrt(stats_df['rvm_std'] ** 2 + (stats_df['rvm_mean'] - overall_rvm_mean) ** 2)

    # Set y-limits for SD plots
    max_std = max(lvm_std_from_overall.max(), rvm_std_from_overall.max())
    y_limit_std = max_std * 1.15

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex='col')
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # Top row: average mass for each patient with overall mean line
    ax1.plot(x_pos, stats_df['lvm_mean'], color='green', marker='o', linewidth=2, label='LVM Mean')
    ax1.axhline(y=overall_lvm_mean, color='darkgreen', linestyle='--', linewidth=2, label=f'Overall Mean: {overall_lvm_mean:.2f}g')
    ax1.set_ylabel('LVM Mean (g)', fontsize=12, fontweight='bold', color='green')
    ax1.set_title('LVM Average Mass Across Patients', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)

    ax2.plot(x_pos, stats_df['rvm_mean'], color='red', marker='o', linewidth=2, label='RVM Mean')
    ax2.axhline(y=overall_rvm_mean, color='darkred', linestyle='--', linewidth=2, label=f'Overall Mean: {overall_rvm_mean:.2f}g')
    ax2.set_ylabel('RVM Mean (g)', fontsize=12, fontweight='bold', color='red')
    ax2.set_title('RVM Average Mass Across Patients', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)

    # Bottom row: SD from overall mean
    bars1 = ax3.bar(x_pos, lvm_std_from_overall, color='green', alpha=0.7, edgecolor='darkgreen', linewidth=1.5)
    ax3.set_xlabel('Patient', fontsize=12, fontweight='bold')
    ax3.set_ylabel('SD from Overall Mean (g)', fontsize=12, fontweight='bold', color='green')
    ax3.set_title('LVM SD from Overall Mean', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0, y_limit_std)

    for bar, val in zip(bars1, lvm_std_from_overall):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * y_limit_std,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    bars2 = ax4.bar(x_pos, rvm_std_from_overall, color='red', alpha=0.7, edgecolor='darkred', linewidth=1.5)
    ax4.set_xlabel('Patient', fontsize=12, fontweight='bold')
    ax4.set_ylabel('SD from Overall Mean (g)', fontsize=12, fontweight='bold', color='red')
    ax4.set_title('RVM SD from Overall Mean', fontsize=13, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim(0, y_limit_std)

    for bar, val in zip(bars2, rvm_std_from_overall):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * y_limit_std,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Shared x-axis labels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats_df['patient'], rotation=45, ha='right')

    plt.tight_layout()

    tracking_plot_path = os.path.join(output_dir, f'std_dev_tracking_{timestamp}.png')
    plt.savefig(tracking_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Mass summary plot saved to: {tracking_plot_path}")

    # Save statistics to CSV
    stats_csv_path = os.path.join(output_dir, f'patient_statistics_{timestamp}.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"✓ Patient statistics saved to: {stats_csv_path}")

    # Print statistics summary
    print(f"\nMass Summary:")
    print(f"  LVM Overall Mean: {overall_lvm_mean:.2f}g")
    print(f"  RVM Overall Mean: {overall_rvm_mean:.2f}g")

print("\nPlot generation complete!")

# Generate violated frames scatter plot
if violated_frame_stats:
    violated_df = pd.DataFrame(violated_frame_stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    x_pos = np.arange(len(violated_df))

    ax1.bar(x_pos, violated_df['lvm_violated_pct'], color='green', alpha=0.7, edgecolor='darkgreen', linewidth=1.2)
    ax1.set_xlabel('Patient', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frames Out of Bounds (%)', fontsize=12, fontweight='bold')
    ax1.set_title('LV Violated Frames (%)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(violated_df['patient'], rotation=45, ha='right')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

    ax2.bar(x_pos, violated_df['rvm_violated_pct'], color='red', alpha=0.7, edgecolor='darkred', linewidth=1.2)
    ax2.set_xlabel('Patient', fontsize=12, fontweight='bold')
    ax2.set_title('RV Violated Frames (%)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(violated_df['patient'], rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

    if 'timestamp' not in locals():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    violated_plot_path = os.path.join(output_dir, f'violated_frames_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(violated_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Violated frames plot saved to: {violated_plot_path}")