import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import argparse
import os
import csv
from pathlib import Path
from datetime import datetime

def setup_argparse():
    """
    Set up command line argument parser for thickness processing.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments with attributes:
        - output_dir: Output directory for CSV and plots
        - model_dir: Model directory containing thickness files
    """
    parser = argparse.ArgumentParser(
        description='Process NIfTI thickness files and generate summary statistics and plots.'
    )
    parser.add_argument(
        '-o', '--output_dir',
        dest='output_dir',
        required=True,
        help='Output directory for generated CSV and plots'
    )
    parser.add_argument(
        '-mdir', '--model_dir',
        dest='model_dir',
        required=False,
        default=None,
        help='Model directory containing thickness NIfTI files'
    )
    parser.add_argument(
        '-d', '--data',
        dest='data',
        required=False,
        default=None,
        help='Path to summary CSV file to generate plots from (skips CSV generation)'
    )
    return parser.parse_args()

def create_output_folder(output_dir):
    """
    Create a timestamped folder within the output directory.
    
    Parameters
    ----------
    output_dir : str
        Base output directory path
    
    Returns
    -------
    str
        Path to the newly created timestamped folder
    """
    # Create timestamp for folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_folder = os.path.join(output_dir, f"thickness_analysis_{timestamp}")
    
    # Create the folder
    os.makedirs(timestamped_folder, exist_ok=True)
    print(f"\n✓ Created output folder: {timestamped_folder}")
    
    return timestamped_folder

def locate_thickness_files(model_dir):
    """
    Recursively search for thickness files in model directory.
    
    Searches for files matching the pattern:
    - rv_thickness_[folder_name]_[frame].nii
    - lv_thickness_[folder_name]_[frame].nii
    
    Parameters
    ----------
    model_dir : str
        Path to the model directory to search
    
    Returns
    -------
    dict
        Dictionary mapping file paths to (thickness_type, folder_name, frame) tuples
        thickness_type: 'rv' or 'lv'
        folder_name: Name of the patient folder
        frame: Frame number
    """
    thickness_files = {}
    
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                # Check for rv_thickness pattern
                if file.startswith('rv_thickness_') and file.count('_') >= 2:
                    parts = file.replace('.nii.gz', '').replace('.nii', '').split('_')
                    if len(parts) >= 4:
                        folder_name = parts[2]
                        frame = parts[3]
                        file_path = os.path.join(root, file)
                        thickness_files[file_path] = ('rv', folder_name, frame)
                        continue
                
                # Check for lv_thickness pattern
                if file.startswith('lv_thickness_') and file.count('_') >= 2:
                    parts = file.replace('.nii.gz', '').replace('.nii', '').split('_')
                    if len(parts) >= 4:
                        folder_name = parts[2]
                        frame = parts[3]
                        file_path = os.path.join(root, file)
                        thickness_files[file_path] = ('lv', folder_name, frame)
                        continue
    
    return thickness_files

class NIfTIAnalyzer:
    """
    Class for analyzing and visualizing NIfTI medical imaging files.
    
    This class loads a NIfTI format thickness map and provides comprehensive analysis
    including statistics, distribution histograms, and spatial visualization of thickness
    values across multiple slices.
    
    Attributes
    ----------
    file_path : str
        Path to the NIfTI file
    data_shape : tuple
        Shape of the NIfTI data (e.g., (256, 256, 120))
    unique_data : numpy.ndarray
        Array of unique values found in the NIfTI data
    affine : numpy.ndarray
        Affine transformation matrix (voxel to mm mapping)
    data : numpy.ndarray
        Raw NIfTI data array
    header : nibabel header
        NIfTI file header information
    
    Examples
    --------
    >>> analyzer = NIfTIAnalyzer("patient1/lv_thickness_patient1_002.nii")
    >>> stats = analyzer.get_statistics()
    >>> analyzer.visualize()
    """
    
    def __init__(self, file_path):
        """
            Initialize NIfTIAnalyzer by loading a NIfTI file.
            
        Parameters
        ----------
        file_path : str
            Absolute path to the NIfTI file (.nii or .nii.gz format)
        """
        self.file_path = file_path
        
        # Load the file
        img = nib.load(file_path)
        self.header = img.header
        self.data = img.get_fdata()
        self.affine = img.affine
        
        # Set attributes
        self.data_shape = self.data.shape
        self.unique_data = np.unique(self.data)
    
    def print_file_info(self):
        """Print detailed information about the NIfTI file."""
        print(f"--- File Inspection: {self.file_path} ---")
        print(f"Shape: {self.data_shape}")
        print(f"Voxel Sizes (mm): {self.header.get_zooms()}")
        print(f"Unique Label Values found: {self.unique_data}")
        print("\nAffine Matrix (Voxel -> MM mapping):")
        print(self.affine)
    
    def get_statistics(self, verbose=True):
        """
        Calculate and return thickness statistics.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print statistics to console. Default is True.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'avg_thickness': Average thickness in mm
            - 'min_thickness': Minimum thickness in mm
            - 'max_thickness': Maximum thickness in mm
            - 'max_location': 3D coordinates of maximum
            - 'min_location': 3D coordinates of minimum
        """
        valid_data = self.data[self.data > 0]  # Exclude 0 and background
        valid_data = valid_data[~np.isnan(valid_data)]  # Exclude NaN values
        
        # Find global max and min thickness
        max_thickness = np.nanmax(self.data)
        min_valid_data = self.data[self.data > 0]
        min_thickness = np.nanmin(min_valid_data)
        avg_thickness = np.nanmean(valid_data)
        
        # Find 3D coordinates of max and min
        max_loc = np.unravel_index(np.nanargmax(self.data), self.data.shape)
        min_loc = np.unravel_index(np.nanargmin(np.ma.masked_where(self.data <= 0, self.data)), self.data.shape)
        
        if verbose:
            print("\nThickness Statistics:")
            print(f"Global Maximum Thickness: {max_thickness:.2f} mm")
            print(f"  Location: X={max_loc[1]}, Y={max_loc[0]}, Slice={max_loc[2]}")
            print(f"\nGlobal Minimum Thickness: {min_thickness:.2f} mm")
            print(f"  Location: X={min_loc[1]}, Y={min_loc[0]}, Slice={min_loc[2]}")
            print(f"\nGlobal Average Thickness: {avg_thickness:.2f} mm")
            
            if len(valid_data) > 0:
                print(f"\nSummary Statistics:")
                print(f"Min thickness: {valid_data.min():.2f} mm")
                print(f"Max thickness: {valid_data.max():.2f} mm")
                print(f"Mean thickness: {valid_data.mean():.2f} mm")    
                print(f"Number of voxels with thickness: {len(valid_data)}")
            else:
                print("No valid thickness data found")
        
        return {
            'avg_thickness': float(avg_thickness),
            'min_thickness': float(min_thickness),
            'max_thickness': float(max_thickness),
            'max_location': max_loc,
            'min_location': min_loc
        }
    
    def visualize(self):
        """Generate and display histogram of thickness distribution."""
        valid_data = self.data[self.data > 0]
        valid_data = valid_data[~np.isnan(valid_data)]
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        ax.hist(valid_data, bins=100, edgecolor='black', alpha=0.7, color='blue')
        ax.set_xlabel('Thickness (mm)')
        ax.set_ylabel('Number of Voxels')
        ax.set_title('Histogram: Distribution of Left Ventricle Wall Thickness')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def inspect_nifti(file_path, return_stats=False, verbose=True):
    """
    Legacy function wrapper for NIfTIAnalyzer class.
    
    This function maintains backward compatibility with existing code.
    For new code, use NIfTIAnalyzer class directly.
    
    Parameters
    ----------
    file_path : str
        Absolute path to the NIfTI file (.nii or .nii.gz format)
    return_stats : bool, optional
        If True, return statistics dictionary. Default is False.
    verbose : bool, optional
        If True, print analysis information. Default is True.
    
    Returns
    -------
    dict or None
        If return_stats is True, returns dictionary with thickness statistics.
        Otherwise, displays visualization and returns None.
    """
    analyzer = NIfTIAnalyzer(file_path)
    
    if verbose:
        analyzer.print_file_info()
        print("\nPlotting thickness distribution...")
    
    if return_stats:
        return analyzer.get_statistics(verbose=verbose)
    else:
        analyzer.get_statistics(verbose=verbose)
        analyzer.visualize()

def generate_summary_csv(thickness_files, output_dir):
    """
    Generate summary CSV file from thickness files.
    
    Parameters
    ----------
    thickness_files : dict
        Dictionary from locate_thickness_files() containing file paths and metadata
    output_dir : str
        Output directory for the CSV file
    
    Returns
    -------
    str
        Path to the generated CSV file
    pandas.DataFrame
        DataFrame containing the summary statistics
    """
    import pandas as pd
    
    summary_data = []
    processed_folders = {}
    completed_printed_folders = set()
    
    for file_path, (thickness_type, folder_name, frame) in thickness_files.items():
        try:
            stats = inspect_nifti(file_path, return_stats=True, verbose=False)
            if stats:
                summary_data.append({
                    'folder_name': folder_name,
                    'frame': frame,
                    'thickness_type': thickness_type,
                    'avg_thickness': stats['avg_thickness'],
                    'min_thickness': stats['min_thickness'],
                    'max_thickness': stats['max_thickness'],
                    'skip_rv': '',
                    'skip_lv': ''
                })
                # Track processed folders
                if folder_name not in processed_folders:
                    processed_folders[folder_name] = set()
                processed_folders[folder_name].add(thickness_type)
                
                # Print completion status as soon as folder is done
                if folder_name not in completed_printed_folders:
                    types_found = processed_folders[folder_name]
                    if 'rv' in types_found and 'lv' in types_found:
                        print(f"✓ Completed {folder_name} (rv, lv)")
                        completed_printed_folders.add(folder_name)
        except Exception as e:
            pass
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV file
    csv_path = os.path.join(output_dir, 'summary_thickness.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary CSV saved to {csv_path}")
    
    return csv_path, df

def generate_sd_plot(summary_df, output_dir):
    """
    Generate standard deviation plots for thickness statistics.
    
    Creates side-by-side plots showing standard deviations of avg_thickness, min_thickness,
    and max_thickness for each folder, separated by RV and LV, with dashed lines 
    indicating the average of each statistic.
    
    Parameters
    ----------
    summary_df : pandas.DataFrame
        DataFrame from generate_summary_csv() with thickness statistics
    output_dir : str
        Output directory for the plot
    
    Returns
    -------
    str
        Path to the saved plot
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    axes = [ax1, ax2]
    thickness_types = ['rv', 'lv']
    
    for idx, thickness_type in enumerate(thickness_types):
        ax = axes[idx]
        
        # Filter data by thickness type
        filtered_df = summary_df[summary_df['thickness_type'] == thickness_type]
        
        if filtered_df.empty:
            continue
        
        # Group by folder_name and calculate statistics
        grouped = filtered_df.groupby('folder_name')[['avg_thickness', 'min_thickness', 'max_thickness']].agg(['mean', 'std'])
        
        # Calculate overall standard deviations for this type
        overall_sd_avg = filtered_df['avg_thickness'].std()
        overall_sd_min = filtered_df['min_thickness'].std()
        overall_sd_max = filtered_df['max_thickness'].std()
        
        folder_names = grouped.index.tolist()
        x_pos = np.arange(len(folder_names))
        width = 0.25
        
        # Extract standard deviations for each metric
        sd_avg = grouped[('avg_thickness', 'std')].values
        sd_min = grouped[('min_thickness', 'std')].values
        sd_max = grouped[('max_thickness', 'std')].values
        
        # Plot bars
        bars1 = ax.bar(x_pos - width, sd_avg, width, label='SD (Avg Thickness)', alpha=0.8)
        bars2 = ax.bar(x_pos, sd_min, width, label='SD (Min Thickness)', alpha=0.8)
        bars3 = ax.bar(x_pos + width, sd_max, width, label='SD (Max Thickness)', alpha=0.8)
        
        # Add dashed lines for overall averages
        ax.axhline(y=overall_sd_avg, color='C0', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg SD (Avg): {overall_sd_avg:.3f}')
        ax.axhline(y=overall_sd_min, color='C1', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg SD (Min): {overall_sd_min:.3f}')
        ax.axhline(y=overall_sd_max, color='C2', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg SD (Max): {overall_sd_max:.3f}')
        
        # Customize plot
        ax.set_xlabel('Folder Name', fontsize=12)
        ax.set_ylabel('Standard Deviation (mm)', fontsize=12)
        ax.set_title(f'Standard Deviation of Thickness Metrics by Folder ({thickness_type.upper()})', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(folder_names, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add secondary y-axis
        ax2 = ax.twinx()
        ax2.set_ylabel('Mean Thickness (mm)', fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Calculate means for secondary axis
        mean_avg = grouped[('avg_thickness', 'mean')].values
        mean_min = grouped[('min_thickness', 'mean')].values
        mean_max = grouped[('max_thickness', 'mean')].values
        
        # Plot lines on secondary axis
        ax2.plot(x_pos - width, mean_avg, 'o-', color='C0', alpha=0.5, linewidth=2, markersize=6, label='Mean (Avg)')
        ax2.plot(x_pos, mean_min, 's-', color='C1', alpha=0.5, linewidth=2, markersize=6, label='Mean (Min)')
        ax2.plot(x_pos + width, mean_max, '^-', color='C2', alpha=0.5, linewidth=2, markersize=6, label='Mean (Max)')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'sd_plot_thickness.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ SD plot saved to {plot_path}")
    
    plt.show()
    return plot_path

def generate_feature_plots(summary_df, output_dir):
    """
    Generate separate plots for each feature (min, max, average thickness).
    
    Creates three individual plots, one for each thickness feature:
    - Average Thickness
    - Minimum Thickness
    - Maximum Thickness
    
    Each plot shows RV and LV data separated by folder name with error bars
    representing the standard deviation.
    
    Parameters
    ----------
    summary_df : pandas.DataFrame
        DataFrame from generate_summary_csv() with thickness statistics
    output_dir : str
        Output directory for the plots
    
    Returns
    -------
    list
        List of paths to the saved plots
    """
    import pandas as pd
    
    features = ['avg_thickness', 'min_thickness', 'max_thickness']
    feature_labels = ['Average Thickness (mm)', 'Minimum Thickness (mm)', 'Maximum Thickness (mm)']
    plot_paths = []
    
    for feature, label in zip(features, feature_labels):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        axes = [ax1, ax2]
        thickness_types = ['rv', 'lv']
        
        for idx, thickness_type in enumerate(thickness_types):
            ax = axes[idx]
            
            # Filter data by thickness type
            filtered_df = summary_df[summary_df['thickness_type'] == thickness_type]
            
            if filtered_df.empty:
                ax.text(0.5, 0.5, f'No data found for {thickness_type.upper()}',
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{label} - {thickness_type.upper()}', fontsize=14, fontweight='bold')
                continue
            
            # Group by folder_name and calculate statistics
            grouped = filtered_df.groupby('folder_name')[feature].agg(['mean', 'std'])
            
            folder_names = grouped.index.tolist()
            x_pos = np.arange(len(folder_names))
            
            # Extract mean and std values
            means = grouped['mean'].values
            stds = grouped['std'].values
            
            # Calculate overall average for this feature
            overall_avg = filtered_df[feature].mean()
            
            # Plot bars without error bars
            bars = ax.bar(x_pos, means, alpha=0.7, 
                         color='steelblue', edgecolor='black', linewidth=1.2)
            
            # Add horizontal line for overall average
            ax.axhline(y=overall_avg, color='red', linestyle='--', linewidth=2, 
                      alpha=0.7, label=f'Overall Avg: {overall_avg:.2f}')
            
            # Customize plot
            ax.set_xlabel('Folder Name', fontsize=12, fontweight='bold')
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(f'{label} - {thickness_type.upper()}', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(folder_names, rotation=45, ha='right')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add secondary y-axis for standard deviation
            ax2 = ax.twinx()
            ax2.set_ylabel('Standard Deviation (mm)', fontsize=12, color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            
            # Plot standard deviation as lines on secondary axis
            ax2.plot(x_pos, stds, 'o-', color='purple', alpha=0.6, linewidth=2, markersize=6, label='Std Dev')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        feature_name = feature.replace('_thickness', '')
        plot_path = os.path.join(output_dir, f'{feature_name}_thickness_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ {label} plot saved to {plot_path}")
        
        plot_paths.append(plot_path)
        plt.close()
    
    return plot_paths

def load_and_generate_plots(csv_path, output_dir):
    """
    Load summary data from CSV and generate all plots (feature plots and SD plot).
    
    This function skips CSV generation and only creates visualizations from existing data.
    Useful for regenerating plots from previously generated summary CSV files.
    
    Supports skip flags: If skip_rv or skip_lv columns contain "*", those rows are excluded
    from plot generation for the corresponding thickness type.
    
    Parameters
    ----------
    csv_path : str
        Path to the summary CSV file (must have same format as summary_thickness.csv)
        Required columns: folder_name, frame, thickness_type, avg_thickness, min_thickness, max_thickness, skip_rv, skip_lv
    output_dir : str
        Output directory for the plots
    
    Returns
    -------
    list
        List of paths to the generated plot files
    """
    import pandas as pd
    
    # Read the CSV file
    try:
        summary_df = pd.read_csv(csv_path)
        print(f"✓ Loaded summary data from {csv_path}")
        print(f"  Found {len(summary_df)} records")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    # Filter out rows based on skip flags (if skip columns exist)
    # For RV: skip if thickness_type=='rv' AND skip_rv contains '*'
    # For LV: skip if thickness_type=='lv' AND skip_lv contains '*'
    original_len = len(summary_df)
    
    # Only apply skip logic if skip columns exist in the CSV
    if 'skip_rv' in summary_df.columns and 'skip_lv' in summary_df.columns:
        skip_mask = pd.Series([False] * len(summary_df), index=summary_df.index)
        for idx, row in summary_df.iterrows():
            if row['thickness_type'] == 'rv' and pd.notna(row.get('skip_rv', '')) and '*' in str(row.get('skip_rv', '')):
                skip_mask.loc[idx] = True
            elif row['thickness_type'] == 'lv' and pd.notna(row.get('skip_lv', '')) and '*' in str(row.get('skip_lv', '')):
                skip_mask.loc[idx] = True
        
        summary_df = summary_df[~skip_mask]
        
        if len(summary_df) < original_len:
            print(f"  Skipped {original_len - len(summary_df)} records based on skip flags")
            print(f"  Generating plots with {len(summary_df)} records")
    else:
        print(f"  Note: CSV does not have skip_rv/skip_lv columns, generating plots with all {len(summary_df)} records")
    
    # Create timestamped output folder (same format as create_output_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_folder = os.path.join(output_dir, f"thickness_analysis_{timestamp}")
    os.makedirs(timestamped_folder, exist_ok=True)
    print(f"\n✓ Created output folder: {timestamped_folder}")
    
    # Generate feature plots
    print("\nGenerating feature plots...")
    feature_plots = generate_feature_plots(summary_df, timestamped_folder)
    
    # Generate SD plot
    print("\nGenerating SD plot...")
    sd_plot = generate_sd_plot(summary_df, timestamped_folder)
    
    return feature_plots + [sd_plot]

def main():
    r"""
    Main function to orchestrate thickness processing pipeline.
    ---
    Two workflow modes:
    
    1. Full pipeline (generate CSV + plots):
       python automate_thickness_processing.py -mdir "path/to/thickness/files" -o "path/to/output"
    
    2. Plots only from existing CSV:
       python automate_thickness_processing.py -d "path/to/summary_thickness.csv" -o "path/to/output"
    
    example usage (use forward slashes for Windows paths in command line):
    >>> python automate_thickness_processing.py -mdir "Z:\sandboxes\Jerry\hpc biv-me data\analysis\Tof_44cases_thickness" -o "C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_thickness"
    
    >>> python automate_thickness_processing.py -d "C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_thickness\thickness_analysis_20260119_142804\summary_thickness.csv" -o "C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_thickness"
    """
    # Parse arguments
    args = setup_argparse()
    
    # Check if using CSV mode or full pipeline mode
    if args.data:
        # Mode 1: Load CSV and generate plots only
        if not os.path.exists(args.data):
            print(f"Error: CSV file not found at {args.data}")
            return
        
        print(f"Using summary data mode...")
        load_and_generate_plots(args.data, args.output_dir)
        print("\n✓ Plot generation complete!")
        
    else:
        # Mode 2: Full pipeline (CSV generation + plots)
        if not args.model_dir:
            print("Error: Either -mdir (for full pipeline) or -d (for plots only) must be specified")
            return
        
        # Create timestamped output folder
        output_folder = create_output_folder(args.output_dir)
        
        # Locate thickness files
        print(f"Searching for thickness files in {args.model_dir}...")
        thickness_files = locate_thickness_files(args.model_dir)
        
        if not thickness_files:
            print("No thickness files found!")
            return
        
        print(f"Found {len(thickness_files)} thickness files")
        
        # Generate summary CSV
        csv_path, summary_df = generate_summary_csv(thickness_files, output_folder)
        
        # Generate individual feature plots
        feature_plots = generate_feature_plots(summary_df, output_folder)
        
        print("\n✓ Processing complete!")

# Run the function
# inspect_nifti(r"C:\Users\jchu579\Documents\SRS 2025_26\biv-me-dev\src\bivme\analysis\example_thickness\patient1\lv_thickness_patient1_000.nii")
# inspect_nifti(r"C:\Users\jchu579\Documents\SRS 2025_26\biv-me-dev\src\bivme\analysis\example_thickness\patient1\lv_thickness_patient1_001.nii")
if __name__ == "__main__":
    main()
