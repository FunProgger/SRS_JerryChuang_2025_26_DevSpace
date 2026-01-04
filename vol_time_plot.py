import pandas as pd 
import matplotlib.pyplot as plt 
import os 

dir = r"C:\Project\SRS\dev_space\analysis"

if dir == None:
    print("Directory is None")

# Get all CSV files from subdirectories
csv_files = []
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

# Warning if no CSV files found
if not csv_files:
    print(f"Warning: No CSV files found in {dir} or its subdirectories")
else:
    print(f"Found {len(csv_files)} CSV file(s)")

for file_path in csv_files:
    df = pd.read_csv(file_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # primary axis for lv_vol
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('lv_vol (mL)')
    ax1.plot(df['frame'], df['lv_vol'], label='lv_vol', color='blue')
    # secondary axis for lvm
    ax3 = ax1.twinx()
    ax3.set_ylabel('lvm (g)')
    ax3.plot(df['frame'], df['lvm'], label='lv_myo_mass', color='green')
    ax1.legend()
    ax1.set_title(f'LV Volume over Frames for {os.path.basename(file_path)}')

    # primary axis for rv_vol
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('rv_vol (mL)')
    ax2.plot(df['frame'], df['rv_vol'], label='rv_vol', color='orange')
    # secondary axis for rvm
    ax4 = ax2.twinx()
    ax4.set_ylabel('rvm (g)')
    ax4.plot(df['frame'], df['rvm'], label='rv_myo_mass', color='red')
    ax2.legend()
    ax2.set_title(f'RV Volume over Frames for {os.path.basename(file_path)}')
    plt.tight_layout()
    
    # Save plot with folder name
    folder_name = os.path.basename(os.path.dirname(file_path))
    output_path = os.path.join(os.path.dirname(file_path), f"{folder_name}_vol_time_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")
