from automate_thickness_processing import NIfTIAnalyzer
from processing_label_image import find_nii_folders
import os
import numpy as np
import matplotlib.pyplot as plt
import maximum_inspect_nifti as m_i_n
import csv

def massData(analyzer: NIfTIAnalyzer): 
    nii_data = analyzer.get_fdata()
    valid_data = nii_data[nii_data > 0]

    heart_density = 1.05  # g/mL
    mm_to_ml = len(valid_data)/1000 # Convert mm^3 to mL
    mass = mm_to_ml * heart_density  # in grams
    return mass

def compute_folder_stats(nii_files_by_folder):
    """
    Compute mass statistics for NIfTI files organized by folder.
    
    Returns
    -------
    dict
        Nested dictionary with folder names as keys and statistics for each file.
    """
    folder_stats = {}
    for folder, files in nii_files_by_folder.items():
        folder_stats[folder] = {}
        for file in files:
            nii = NIfTIAnalyzer(file_path=os.path.join(folder, file))
            mass_series = massData(nii)

            avg_mass = np.mean(mass_series)
            std_mass = np.std(mass_series)

            folder_stats[folder][file] = {
                "masses": mass_series,
                "avg_mass": float(avg_mass),
                "std_mass": float(std_mass),
            }

        print(f"✓ Complete mass stats for {folder}")

    return folder_stats

def extract_frame_from_filename(filename: str):
    # Frame extraction happens here.
    # Expected filename pattern: lv_thickness_CASEXXXX_XXX.nii(.gz)
    # Example: lv_thickness_CHD0004001_007.nii -> frame = 7
    name = filename
    if name.endswith('.nii.gz'):
        name = name[:-7]
    elif name.endswith('.nii'):
        name = name[:-4]

    parts = name.split('_')
    if not parts:
        return None

    # Frame is the last underscore-separated token
    frame_str = parts[-1]
    try:
        return int(frame_str)
    except ValueError:
        return None

def plot_folder_masses(folder_stats, nii_files_by_folder, save_path):
    for folder, files in nii_files_by_folder.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.set_title(f"Voxel to Mass: LV Mass over Frames for {os.path.basename(folder)}")
        ax1.set_xlabel("Frames")
        ax1.set_ylabel("Mass (g)")

        ax2.set_title(f"Voxel to Mass: RV Mass over Frames for {os.path.basename(folder)}")
        ax2.set_xlabel("Frames")
        ax2.set_ylabel("Mass (g)")

        lv_points = []
        rv_points = []
        # Use extracted frame as x-axis, and avg_mass (per file) as y-axis
        for file in files:
            frame = extract_frame_from_filename(file)
            if frame is None:
                continue
            avg_mass = folder_stats[folder][file]["avg_mass"]

            if "lv" in file.lower():
                lv_points.append((frame, avg_mass, os.path.basename(file)))
            elif "rv" in file.lower():
                rv_points.append((frame, avg_mass, os.path.basename(file)))

        if lv_points:
            lv_points.sort(key=lambda x: x[0])
            lv_frames = [p[0] for p in lv_points]
            lv_masses = [p[1] for p in lv_points]
            ax1.plot(lv_frames, lv_masses, color='blue', marker='o', label='LV Mass')

        if rv_points:
            rv_points.sort(key=lambda x: x[0])
            rv_frames = [p[0] for p in rv_points]
            rv_masses = [p[1] for p in rv_points]
            ax2.plot(rv_frames, rv_masses, color='red', marker='o', label='RV Mass')

        ax1.legend(fontsize=9)
        ax2.legend(fontsize=9)

        output_path = os.path.join(save_path, f"{os.path.basename(folder)}_mass_over_frames.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ {folder} Mass plot saved: {output_path}")

def write_mass_csv(folder_stats, nii_files_by_folder, save_path, filename="mass_summary.csv"):
    output_path = os.path.join(save_path, filename)
    rows = []

    for folder, files in nii_files_by_folder.items():
        folder_name = os.path.basename(folder)
        lv_by_frame = {}
        rv_by_frame = {}

        for file in files:
            frame = extract_frame_from_filename(file)
            if frame is None:
                continue
            avg_mass = folder_stats[folder][file]["avg_mass"]

            if "lv" in file.lower():
                lv_by_frame[frame] = avg_mass
            elif "rv" in file.lower():
                rv_by_frame[frame] = avg_mass

        all_frames = sorted(set(lv_by_frame.keys()) | set(rv_by_frame.keys()))
        for frame in all_frames:
            rows.append({
                "foldername": folder_name,
                "frame": frame,
                "lvm": lv_by_frame.get(frame, None),
                "rvm": rv_by_frame.get(frame, None),
            })

    with open(output_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["foldername", "frame", "lvm", "rvm"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Mass CSV saved: {output_path}")

        

if __name__ == "__main__":
    
    mdir = r"Z:\sandboxes\Jerry\hpc biv-me data\analysis\Tof_44cases_thickness"
    odir = r"Z:\sandboxes\Jerry\hpc biv-me data\analysis\Voxel_Mass"
    nii_folders = find_nii_folders(mdir)

    nii_files_by_folder = {
            folder: [
                f for f in os.listdir(folder)
                if (f.endswith('.nii') or f.endswith('.nii.gz'))
                and "labeled" not in f
            ]
            for folder in nii_folders
        }

    output_dir = m_i_n.create_output_folder_from_model_dir(odir, mdir)
    folder_stats = compute_folder_stats(nii_files_by_folder)

    # Plot mass over frames for each folder
    plot_folder_masses(folder_stats, nii_files_by_folder, output_dir)

    # Save CSV summary
    write_mass_csv(folder_stats, nii_files_by_folder, output_dir)









