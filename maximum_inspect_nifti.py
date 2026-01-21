import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from automate_thickness_processing import NIfTIAnalyzer, locate_thickness_files, create_output_folder

def setup_argparse():
    """
    Set up command line argument parser for maximum thickness inspection.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments with attributes:
        - mdir: Path to the directory containing NIfTI files to inspect
        - odir: Output directory for visualizations
    """
    parser = argparse.ArgumentParser(
        description='Inspect and analyze maximum thickness from NIfTI files.'
    )
    parser.add_argument(
        '-mdir', '--model_dir',
        dest='mdir',
        required=True,
        help='Path to the directory containing NIfTI files to inspect'
    )
    parser.add_argument(
        '-odir', '--output_dir',
        dest='odir',
        required=True,
        help='Output directory for visualizations'
    )
    return parser.parse_args()

def detect_and_visualize_outliers(file_path, output_dir, threshold=50):
    """
    Detect thickness values above threshold and generate heatmap visualizations.
    
    For each slice containing outlier maximum values, creates a heatmap showing
    the thickness distribution with the maximum location marked.
    
    Parameters
    ----------
    file_path : str
        Path to the NIfTI thickness file
    output_dir : str
        Output directory for the visualization images
    threshold : float, optional
        Threshold value for detecting outliers. Default is 50 mm.
    
    Returns
    -------
    list
        List of paths to generated visualization images
    """
    analyzer = NIfTIAnalyzer(file_path)
    stats = analyzer.get_statistics(verbose=False)
    
    viz_paths = []
    
    # Check if maximum exceeds threshold
    if stats['max_thickness'] > threshold:
        max_loc = stats['max_location']
        slice_idx = max_loc[2]
        y_coord = max_loc[0]
        x_coord = max_loc[1]
        
        # Extract the slice containing the maximum
        slice_data = analyzer.data[:, :, slice_idx]
        
        # Create heatmap visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Use logarithmic scale if maximum exceeds 100
        if stats['max_thickness'] > 100:
            # Filter out zero and negative values for log scale
            slice_data_positive = slice_data.copy()
            slice_data_positive[slice_data_positive <= 0] = np.nan
            
            im = ax.imshow(slice_data_positive, cmap='hot', origin='lower', aspect='auto', 
                          norm=LogNorm(vmin=np.nanmin(slice_data_positive), vmax=np.nanmax(slice_data_positive)))
            scale_type = "Log Scale"
        else:
            im = ax.imshow(slice_data, cmap='hot', origin='lower', aspect='auto')
            scale_type = "Linear Scale"
        
        # Mark the location of maximum thickness with red X
        ax.plot(x_coord, y_coord, 'rx', markersize=15, markeredgewidth=3, label='Max Thickness Location')
        
        # Extract folder name and frame from file path
        file_name = os.path.basename(file_path)
        parts = file_name.replace('.nii.gz', '').replace('.nii', '').split('_')
        thickness_type = parts[0]  # 'rv' or 'lv'
        folder_name = parts[2] if len(parts) > 2 else 'unknown'
        frame = parts[3] if len(parts) > 3 else '00'
        
        # Set title and labels
        ax.set_title(f'Outlier Heatmap: {thickness_type.upper()} - {folder_name} (Frame {frame})\n'
                    f'Slice {slice_idx} | Max Thickness: {stats["max_thickness"]:.2f} mm | {scale_type}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=10)
        ax.set_ylabel('Y Coordinate', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Thickness (mm)', fontsize=10)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', color='white')
        
        plt.tight_layout()
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        viz_filename = f'outlier_heatmap_{folder_name}_{frame}_{thickness_type}.png'
        viz_path = os.path.join(output_dir, viz_filename)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Outlier heatmap saved: {viz_path}")
        
        viz_paths.append(viz_path)
        plt.close()
    
    return viz_paths

def main():
    args = setup_argparse()
    
    output_dir = create_output_folder(args.odir)
    nifti_list = locate_thickness_files(args.mdir)
    
    for nifti in nifti_list:
        analyzer = NIfTIAnalyzer(nifti) 
        stats = analyzer.get_statistics(verbose=False)
        
        # Detect and visualize outliers (threshold = 50 mm)
        check_thickness = 0 
        if stats['max_thickness'] > check_thickness:
            detect_and_visualize_outliers(nifti, output_dir, threshold=check_thickness)
    

if __name__ == "__main__":
    main()