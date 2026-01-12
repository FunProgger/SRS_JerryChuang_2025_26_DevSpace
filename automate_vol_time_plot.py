"""
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
- Generates a separate cross-patient standard deviation tracking plot
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

Standard Deviation Tracking Plot (PNG)
    Separate image file (std_dev_tracking_{timestamp}.png) showing:
    - Left panel: LVM standard deviation for each patient (green bars)
    - Right panel: RVM standard deviation for each patient (red bars)
    - Mean lines showing average standard deviation across all patients
    - Value labels on each bar for precise tracking

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

    python automate_vol_time_plot.py -mdir "C:/Users/jchu579/Documents/SRS 2025_26/bivme-data/analysis/Vol_Time_plots_after_troubleshoot"

With custom output directory::

    python automate_vol_time_plot.py -mdir C:/Users/jchu579/Documents/SRS 2025_26/bivme-data -o C:/output/path
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import argparse
import numpy as np
from datetime import datetime

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
    output_dir = os.path.join(dir, f"lvrv_volume_plots_{timestamp}")
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
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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
                
                # Plot variation from mean (on the left, offset)
                lvm_deviation = patient_df['lvm'] - lvm_mean
                ax3_dev = ax1.twinx()
                ax3_dev.spines['left'].set_position(('outward', 60))
                ax3_dev.set_ylabel('LVM Deviation from Mean (g)', fontsize=10, color='darkgreen')
                ax3_dev.plot(patient_df['frame'], lvm_deviation, color='darkgreen', linewidth=1.5, alpha=0.7, linestyle='-')
                ax3_dev.scatter(patient_df['frame'], lvm_deviation, color='darkgreen', s=50, alpha=0.6, 
                               marker='^', label='lvm_deviation', edgecolors='darkgreen', linewidth=1)
                ax3_dev.tick_params(axis='y', labelcolor='darkgreen')
                ax3_dev.yaxis.set_label_position('left')
                ax3_dev.yaxis.tick_left()
                ax3_dev.axhline(y=0, color='darkgreen', linestyle=':', linewidth=1, alpha=0.5)
                
                # Combined legend
                lines = line1 + line2
                lines.append(line_avg_lvm)
                labels = [l.get_label() for l in lines[:-1]] + ['lvm_avg', 'lvm_error_band', 'lvm_deviation']
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
                
                # Plot variation from mean (on the left, offset)
                rvm_deviation = patient_df['rvm'] - rvm_mean
                ax4_dev = ax2.twinx()
                ax4_dev.spines['left'].set_position(('outward', 60))
                ax4_dev.set_ylabel('RVM Deviation from Mean (g)', fontsize=10, color='darkred')
                ax4_dev.plot(patient_df['frame'], rvm_deviation, color='darkred', linewidth=1.5, alpha=0.7, linestyle='-')
                ax4_dev.scatter(patient_df['frame'], rvm_deviation, color='darkred', s=50, alpha=0.6, 
                               marker='^', label='rvm_deviation', edgecolors='darkred', linewidth=1)
                ax4_dev.tick_params(axis='y', labelcolor='darkred')
                ax4_dev.yaxis.set_label_position('left')
                ax4_dev.yaxis.tick_left()
                ax4_dev.axhline(y=0, color='darkred', linestyle=':', linewidth=1, alpha=0.5)
                
                # Combined legend
                lines = line3 + line4
                lines.append(line_avg_rvm)
                labels = [l.get_label() for l in lines[:-1]] + ['rvm_avg', 'rvm_error_band', 'rvm_deviation']
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

# Generate cross-patient standard deviation tracking plot
if patient_stats:
    stats_df = pd.DataFrame(patient_stats)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # LVM standard deviation plot
    x_pos = np.arange(len(stats_df))
    bars1 = ax1.bar(x_pos, stats_df['lvm_std'], color='green', alpha=0.7, edgecolor='darkgreen', linewidth=1.5)
    ax1.set_xlabel('Patient', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Standard Deviation (g)', fontsize=12, fontweight='bold', color='green')
    ax1.set_title('LVM Standard Deviation Across All Patients', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stats_df['patient'], rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, stats_df['lvm_std'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * stats_df['lvm_std'].max(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add mean line
    lvm_std_mean = stats_df['lvm_std'].mean()
    lvm_std_overall = stats_df['lvm_std'].std()
    ax1.axhline(y=lvm_std_mean, color='darkgreen', linestyle='--', linewidth=2, 
               label=f'Mean: {lvm_std_mean:.2f}g', alpha=0.8)
    ax1.legend(fontsize=10)
    
    # RVM standard deviation plot
    bars2 = ax2.bar(x_pos, stats_df['rvm_std'], color='red', alpha=0.7, edgecolor='darkred', linewidth=1.5)
    ax2.set_xlabel('Patient', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Standard Deviation (g)', fontsize=12, fontweight='bold', color='red')
    ax2.set_title('RVM Standard Deviation Across All Patients', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stats_df['patient'], rotation=45, ha='right')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, stats_df['rvm_std'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * stats_df['rvm_std'].max(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add mean line
    rvm_std_mean = stats_df['rvm_std'].mean()
    rvm_std_overall = stats_df['rvm_std'].std()
    ax2.axhline(y=rvm_std_mean, color='darkred', linestyle='--', linewidth=2, 
               label=f'Mean: {rvm_std_mean:.2f}g', alpha=0.8)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the tracking plot
    tracking_plot_path = os.path.join(output_dir, f'std_dev_tracking_{timestamp}.png')
    plt.savefig(tracking_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Standard deviation tracking plot saved to: {tracking_plot_path}")
    
    # Save statistics to CSV
    stats_csv_path = os.path.join(output_dir, f'patient_statistics_{timestamp}.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"✓ Patient statistics saved to: {stats_csv_path}")
    
    # Print statistics summary
    print(f"\nStandard Deviation Summary:")
    print(f"  LVM Std Dev - Mean: {lvm_std_mean:.2f}g, Overall Std: {lvm_std_overall:.2f}g")
    print(f"  RVM Std Dev - Mean: {rvm_std_mean:.2f}g, Overall Std: {rvm_std_overall:.2f}g")

print("\nPlot generation complete!")