import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

def inspect_nifti(file_path):
    # 1. Load the file
    img = nib.load(file_path)
    header = img.header
    data = img.get_fdata()
    affine = img.affine

    print(f"--- File Inspection: {file_path} ---")
    
    # 2. Dimensions and Resolution
    print(f"Shape: {data.shape}") # e.g., (256, 256, 120)
    print(f"Voxel Sizes (mm): {header.get_zooms()}")
    
    # 3. Label Check
    unique_labels = np.unique(data)
    print(f"Unique Label Values found: {unique_labels}")
    
    # Print full array without truncation
    # np.set_printoptions(threshold=np.inf, suppress=True)
    # print(f"\nFull Unique Values:\n{unique_labels}")
    
    # 4. Spatial Orientation (The Affine)
    print("\nAffine Matrix (Voxel -> MM mapping):")
    print(affine)

    # 5. Plot All Thickness Data
    print("\nPlotting thickness distribution...")
    valid_data = data[data > 0]  # Exclude 0 and background
    valid_data = valid_data[~np.isnan(valid_data)]  # Exclude NaN values
    
    # Find global max and min thickness
    max_thickness = np.nanmax(data)
    min_valid_data = data[data > 0]
    min_thickness = np.nanmin(min_valid_data)
    avg_thickness = np.nanmean(valid_data)
    
    # Find 3D coordinates of max and min
    max_loc = np.unravel_index(np.nanargmax(data), data.shape)
    min_loc = np.unravel_index(np.nanargmin(np.ma.masked_where(data <= 0, data)), data.shape)
    
    # Find 3D coordinate of average thickness
    avg_loc = np.unravel_index(np.nanargmin(np.abs(data - avg_thickness)), data.shape)
    
    max_slice = max_loc[2]
    min_slice = min_loc[2]
    avg_slice = avg_loc[2]
    
    print(f"\nGlobal Maximum Thickness: {max_thickness:.2f} mm")
    print(f"  Location: X={max_loc[1]}, Y={max_loc[0]}, Slice={max_slice}")
    print(f"\nGlobal Minimum Thickness: {min_thickness:.2f} mm")
    print(f"  Location: X={min_loc[1]}, Y={min_loc[0]}, Slice={min_slice}")
    print(f"\nGlobal Average Thickness: {avg_thickness:.2f} mm")
    print(f"  Location: X={avg_loc[1]}, Y={avg_loc[0]}, Slice={avg_slice}")
    
    
    # Create a figure with custom grid layout
    # Top row: Histogram (left) | Middle slice (right)
    # Bottom row: Min slice (left) | Max slice (middle) | Average slice (right)
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Histogram - top left
    ax2 = fig.add_subplot(gs[0, 1:])  # Middle slice - top right (spans 2 columns)
    ax3 = fig.add_subplot(gs[1, 0])  # Min slice - bottom left
    ax4 = fig.add_subplot(gs[1, 1])  # Max slice - bottom middle
    ax5 = fig.add_subplot(gs[1, 2])  # Average slice - bottom right
    
    
    # Histogram
    ax1.hist(valid_data, bins=100, edgecolor='black', alpha=0.7, color='blue')
    ax1.set_xlabel('Thickness (mm)')
    ax1.set_ylabel('Number of Voxels')
    ax1.set_title('Histogram: Distribution of Left Ventricle Wall Thickness')
    ax1.grid(True, alpha=0.3)
    
    # Middle slice with min, max, and average
    mid_slice = data.shape[2] // 2
    mid_slice_data = data[:, :, mid_slice]
    
    # Find min and max in middle slice
    mid_slice_valid = mid_slice_data[mid_slice_data > 0]
    mid_max = np.nanmax(mid_slice_data)
    mid_min = np.nanmin(mid_slice_valid)
    mid_avg = np.nanmean(mid_slice_valid)
    
    # Find locations in middle slice
    mid_max_loc = np.unravel_index(np.nanargmax(mid_slice_data), mid_slice_data.shape)
    mid_min_loc = np.unravel_index(np.nanargmin(np.ma.masked_where(mid_slice_data <= 0, mid_slice_data)), mid_slice_data.shape)
    
    # Find closest voxel to average
    mid_avg_loc = np.unravel_index(np.nanargmin(np.abs(mid_slice_data - mid_avg)), mid_slice_data.shape)
    
    im_mid = ax2.imshow(mid_slice_data, cmap='viridis', origin='lower')
    ax2.set_xlabel('X (voxels)')
    ax2.set_ylabel('Y (voxels)')
    ax2.set_title(f'Middle Slice {mid_slice} with Min, Max, and Average')
    ax2.plot(mid_max_loc[1], mid_max_loc[0], 'r^', markersize=8, label=f'Max: {mid_max:.2f} mm')
    ax2.plot(mid_min_loc[1], mid_min_loc[0], 'bv', markersize=8, label=f'Min: {mid_min:.2f} mm')
    ax2.plot(mid_avg_loc[1], mid_avg_loc[0], 'gs', markersize=8, label=f'Avg: {mid_avg:.2f} mm')
    ax2.legend(loc='upper right')
    cbar_mid = plt.colorbar(im_mid, ax=ax2)
    cbar_mid.set_label('Thickness (mm)')
    
    # Heatmap of max slice
    max_slice_data = data[:, :, max_slice]
    im2 = ax3.imshow(max_slice_data, cmap='viridis', origin='lower')
    ax3.set_xlabel('X (voxels)')
    ax3.set_ylabel('Y (voxels)')
    ax3.set_title(f'Max Thickness at Slice {max_slice}')
    ax3.plot(max_loc[1], max_loc[0], 'r^', markersize=8, label=f'Max: {max_thickness:.2f} mm')
    ax3.legend(loc='upper right')
    cbar2 = plt.colorbar(im2, ax=ax3)
    cbar2.set_label('Thickness (mm)')
    
    # Heatmap of min slice
    min_slice_data = data[:, :, min_slice]
    im3 = ax4.imshow(min_slice_data, cmap='viridis', origin='lower')
    ax4.set_xlabel('X (voxels)')
    ax4.set_ylabel('Y (voxels)')
    ax4.set_title(f'Min Thickness at Slice {min_slice}')
    ax4.plot(min_loc[1], min_loc[0], 'bv', markersize=8, label=f'Min: {min_thickness:.2f} mm')
    ax4.legend(loc='upper right')
    cbar3 = plt.colorbar(im3, ax=ax4)
    cbar3.set_label('Thickness (mm)')
    
    # Heatmap of average slice
    avg_slice_data = data[:, :, avg_slice]
    im4 = ax5.imshow(avg_slice_data, cmap='viridis', origin='lower')
    ax5.set_xlabel('X (voxels)')
    ax5.set_ylabel('Y (voxels)')
    ax5.set_title(f'Average Thickness at Slice {avg_slice}')
    ax5.plot(avg_loc[1], avg_loc[0], 'gs', markersize=8, label=f'Avg: {avg_thickness:.2f} mm')
    ax5.legend(loc='upper right')
    cbar4 = plt.colorbar(im4, ax=ax5)
    cbar4.set_label('Thickness (mm)')
    
    print(f"\nMiddle Slice {mid_slice} Statistics:")
    print(f"  Max: {mid_max:.2f} mm at X={mid_max_loc[1]}, Y={mid_max_loc[0]}")
    print(f"  Min: {mid_min:.2f} mm at X={mid_min_loc[1]}, Y={mid_min_loc[0]}")
    print(f"  Average: {mid_avg:.2f} mm at X={mid_avg_loc[1]}, Y={mid_avg_loc[0]}")
    
    
    plt.tight_layout()
    plt.show()
    
    # 6. Thickness Statistics
    if len(valid_data) > 0:
        print(f"Min thickness: {valid_data.min():.2f} mm")
        print(f"Max thickness: {valid_data.max():.2f} mm")
        print(f"Mean thickness: {valid_data.mean():.2f} mm")    
        print(f"Number of voxels with thickness: {len(valid_data)}")
    else:
        print("No valid thickness data found")

# Run the function
# inspect_nifti("C:\Users\\jchu579\\Documents\\SRS 2025_26\\biv-me-dev\\src\\bivme\\analysis\\example_thickness\\patient1\\lv_thickness_patient1_000.nii")
# inspect_nifti("C:\\Users\\jchu579\\Documents\\SRS 2025_26\\biv-me-dev\\src\\bivme\\analysis\\example_thickness\\patient1\\lv_thickness_patient1_001.nii")
inspect_nifti("C:\\Users\\jchu579\\Documents\\SRS 2025_26\\biv-me-dev\\src\\bivme\\analysis\\example_thickness\\patient1\\lv_thickness_patient1_002.nii")
