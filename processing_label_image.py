from automate_thickness_processing import NIfTIAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import os
from pathlib import Path
import argparse
import csv
import pandas as pd

def find_nii_folders(root_dir):
    """
    Recursively find all folders that contain .nii or .nii.gz files.
    
    Parameters
    ----------
    root_dir : str
        Root directory to search
    
    Returns
    -------
    list
        List of folder paths that contain .nii files
    """
    folders_with_nii = set()
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                folders_with_nii.add(root)
                break
    
    return sorted(list(folders_with_nii))

def create_heatmap(nii_file_path, output_dir=None, slice_idx=None, slice_lookup=None):
    """
    Create a heatmap visualization from a NIfTI file.
    
    Parameters
    ----------
    nii_file_path : str
        Path to the .nii file
    output_dir : str, optional
        Directory to save the heatmap. If None, displays only.
    slice_idx : int, optional
        Slice index to visualize. If None, uses lookup or middle slice.
    slice_lookup : dict, optional
        Dictionary mapping (case_name, thickness_type, frame) to slice index
    
    Returns
    -------
    str or None
        Path to saved heatmap if output_dir provided, else None
    """
    # Load NIfTI file
    img = nib.load(nii_file_path)
    data = img.get_fdata()
    print(f"get_fdata() shape: {data.shape}")

    x_coord = None
    y_coord = None
    
    # Extract case name and thickness type from filename
    file_name = os.path.basename(nii_file_path)
    
    # Try to parse filename to get case_name, thickness_type, and frame
    x_coord = None
    y_coord = None

    if slice_lookup is not None:
        # Expected format: [thickness_type]_thickness_[case_name]_[frame].nii
        # or labeled_image_[lv/rv]_[case_name]_[frame].nii
        parts = file_name.replace('.nii.gz', '').replace('.nii', '').split('_')
        
        case_name = None
        thickness_type = None
        frame = None
        
        # Pattern 1: lv_thickness_CASEXXXX_XXX or rv_thickness_CASEXXXX_XXX
        if len(parts) >= 4 and parts[1] == 'thickness':
            thickness_type = parts[0].upper()  # LV or RV
            case_name = parts[2]
            frame = parts[3]
        # Pattern 2: labeled_image_lv_CASEXXXX_XXX or labeled_image_rv_CASEXXXX_XXX
        elif len(parts) >= 5 and parts[0] == 'labeled' and parts[1] == 'image':
            thickness_type = parts[2].upper()  # LV or RV
            case_name = parts[3]
            frame = parts[4]
        
        # Look up slice and coordinates from dictionary
        if case_name and thickness_type and frame:
            lookup_key = (case_name, thickness_type, frame)
            if lookup_key in slice_lookup:
                entry = slice_lookup[lookup_key]
                slice_idx = entry['slice']
                x_coord = entry.get('x')
                y_coord = entry.get('y')
                print(f"  Using slice {slice_idx} from CSV for {case_name} {thickness_type} frame {frame}")
            else:
                print(f"  ⚠ Skipping: {case_name} {thickness_type} frame {frame} not found in CSV")
                return None
    
    # If slice_idx is still None (no CSV or lookup failed), skip this file
    if slice_idx is None:
        return None
    
    slice_data = data[:, :, slice_idx]
    
    # Create discrete colormap with 4 colors
    colors = ['blue', 'green', 'yellow', 'red']
    cmap_discrete = ListedColormap(colors)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slice_data, cmap=cmap_discrete, origin='lower', aspect='auto', vmin=-1, vmax=2)

    # Mark max location if coordinates are available (note: data is transposed for display)
    if x_coord is not None and y_coord is not None:
        ax.plot(x_coord, y_coord, 'k.', markersize=5, label='Max Thickness Location')
        ax.legend(loc='upper right', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[-0.625, 0.125, 0.875, 1.625])
    cbar.ax.set_yticklabels(['-1', '0', '1', '2'])
    cbar.set_label('Intensity', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    file_name = os.path.basename(nii_file_path)
    ax.set_title(f'Heatmap: {file_name} (Slice {slice_idx})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{Path(file_name).stem}_heatmap_slice{slice_idx}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None

def process_folder(folder_path, output_base_dir=None, slice_lookup=None):
    """
    Process all .nii files in a folder and create heatmaps.
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing .nii files
    output_base_dir : str, optional
        Base directory for saving heatmaps. Creates subfolder per input folder.
    slice_lookup : dict, optional
        Dictionary mapping (case_name, thickness_type, frame) to slice index
    
    Returns
    -------
    list
        List of paths to generated heatmaps
    """
    nii_files = [f for f in os.listdir(folder_path) 
                 if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    if not nii_files:
        return []
    
    print(f"\nProcessing folder: {folder_path}")
    print(f"  Found {len(nii_files)} .nii files")
    
    heatmap_paths = []
    skipped_files = []
    
    # Create output directory if saving
    if output_base_dir:
        folder_name = os.path.basename(folder_path)
        output_dir = os.path.join(output_base_dir, f"{folder_name}_heatmaps")
    else:
        output_dir = None
    
    for nii_file in nii_files:
        nii_path = os.path.join(folder_path, nii_file)
        try:
            heatmap_path = create_heatmap(nii_path, output_dir, slice_lookup=slice_lookup)
            if heatmap_path:
                heatmap_paths.append(heatmap_path)
                print(f"  ✓ Created heatmap for {nii_file}")
            else:
                skipped_files.append(nii_file)
        except Exception as e:
            print(f"  ✗ Error processing {nii_file}: {e}")
    
    return heatmap_paths, skipped_files

def main():
    """
    Main function to process .nii files and generate heatmaps.
    
    Example usage:
    >>> python processing_label_image.py -i "path/to/root/folder" -o "path/to/output" -csv "path/to/slice_info.csv"
    >>> python processing_label_image.py -i "Z:\\data\\nii_files" -o "C:\\output\\heatmaps" -csv "slices.csv"

    Usage:
    python processing_label_image.py -i "Z:\\sandboxes\\Jerry\\hpc biv-me data\\analysis\\Labelled" -o "C:\\Users\\jchu579\\Documents\\SRS 2025_26\\dev_space\\output_label" -csv "Z:\sandboxes\Jerry\hpc biv-me data\analysis\version1_thickness_rerun\version1_thickness_rerun_thickness_analysis_20260130_144632\maximum_thickness_outliers.csv"
    """
    parser = argparse.ArgumentParser(
        description='Process NIfTI files and generate heatmaps.'
    )
    parser.add_argument(
        '-i', '--input',
        dest='input_dir',
        required=True,
        help='Root directory containing .nii files (can be nested)'
    )
    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        required=False,
        default=None,
        help='Output directory for heatmaps (optional, shows plots if not provided)'
    )
    parser.add_argument(
        '-csv', '--csv_file',
        dest='csv_file',
        required=False,
        default=None,
        help='CSV file with columns: Case Name, thickness_type, frame, maximum value, x, y, slice'
    )
    
    args = parser.parse_args()
    
    # Load slice lookup from CSV if provided
    slice_lookup = None
    if args.csv_file:
        if os.path.exists(args.csv_file):
            try:
                df = pd.read_csv(args.csv_file, dtype={'frame': str})
                # Create lookup dictionary: (case_name, thickness_type, frame) -> slice
                slice_lookup = {}
                for _, row in df.iterrows():
                    case_name = str(row['Case Name']).strip()
                    thickness_type = str(row['thickness_type']).strip().upper()
                    frame = str(row['frame']).strip()
                    slice_idx = int(row['slice'])
                    x_coord = int(row['x']) if 'x' in row and not pd.isna(row['x']) else None
                    y_coord = int(row['y']) if 'y' in row and not pd.isna(row['y']) else None
                    slice_lookup[(case_name, thickness_type, frame)] = {
                        'slice': slice_idx,
                        'x': x_coord,
                        'y': y_coord
                    }
                
                print(f"✓ Loaded slice information from CSV: {args.csv_file}")
                print(f"  Found {len(slice_lookup)} case/type/frame combinations")
            except Exception as e:
                print(f"✗ Error reading CSV file: {e}")
                print("  Continuing with middle slice as default...")
        else:
            print(f"✗ CSV file not found: {args.csv_file}")
            print("  Continuing with middle slice as default...")
    
    # Find all folders containing .nii files
    print(f"\nSearching for .nii files in: {args.input_dir}")
    nii_folders = find_nii_folders(args.input_dir)
    
    if not nii_folders:
        print("No folders with .nii files found!")
        return
    
    print(f"\nFound {len(nii_folders)} folder(s) containing .nii files:")
    for folder in nii_folders:
        print(f"  - {folder}")
    
    # Process each folder
    all_heatmaps = []
    all_skipped = []
    used_csv_entries = set()
    
    # Track which CSV entries are used during processing
    if slice_lookup:
        original_slice_lookup = slice_lookup.copy()
    
    for folder in nii_folders:
        heatmaps, skipped = process_folder(folder, args.output_dir, slice_lookup=slice_lookup)
        all_heatmaps.extend(heatmaps)
        all_skipped.extend(skipped)
    
    # Determine which CSV entries were actually used
    if slice_lookup and args.output_dir:
        # Extract used entries from heatmap filenames
        for heatmap_path in all_heatmaps:
            filename = os.path.basename(heatmap_path)
            # Parse filename to extract case, type, frame
            parts = filename.replace('_heatmap_slice', '_').split('_')
            if len(parts) >= 5 and parts[0] == 'labeled' and parts[1] == 'image':
                thickness_type = parts[2].upper()
                case_name = parts[3]
                frame = parts[4].split('.')[0]  # Remove .png
                used_csv_entries.add((case_name, thickness_type, frame))
        
        unused_csv_entries = set(original_slice_lookup.keys()) - used_csv_entries
    
    # Print summary
    if args.output_dir:
        print(f"\n✓ Processing complete! Generated {len(all_heatmaps)} heatmaps")
        print(f"  Saved to: {args.output_dir}")
    else:
        print(f"\n✓ Processing complete! Displayed {len(all_heatmaps)} heatmaps")
    
    # Report skipped files
    if all_skipped:
        print(f"\n⚠ Skipped {len(all_skipped)} file(s) not found in CSV:")
        for skipped in all_skipped[:10]:  # Show first 10
            print(f"  - {skipped}")
        if len(all_skipped) > 10:
            print(f"  ... and {len(all_skipped) - 10} more")
    
    # Report unused CSV entries
    if slice_lookup and unused_csv_entries:
        print(f"\n⚠ {len(unused_csv_entries)} CSV entry(ies) not processed (no matching file found):")
        for entry in sorted(list(unused_csv_entries))[:10]:  # Show first 10
            case_name, thickness_type, frame = entry
            print(f"  - {case_name} {thickness_type} frame {frame}")
        if len(unused_csv_entries) > 10:
            print(f"  ... and {len(unused_csv_entries) - 10} more")

if __name__ == "__main__":
    main()


