from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

def merge_mass_data(output_csv: str, voxelMass_csv: str, volMass_csv: str): 
    """
    Docstring for merge_mass_data
    TODO: 
    Case by case 
    each case has different number of frames 
    """

    # Load voxel mass data
    voxel_mass_df = pd.read_csv(voxelMass_csv)
    vol_mass_df = pd.read_csv(volMass_csv)

    # Normalize key columns
    if "foldername" in voxel_mass_df.columns:
        voxel_mass_df = voxel_mass_df.rename(columns={"foldername": "name"})
    if "name" not in voxel_mass_df.columns:
        raise ValueError("voxel_mass_df must have 'foldername' or 'name' column")

    if "name" not in vol_mass_df.columns:
        raise ValueError("vol_mass_df must have 'name' column")

    if "frame" in voxel_mass_df.columns:
        voxel_mass_df = voxel_mass_df.rename(columns={"frame": "Frame"})
    if "frame" in vol_mass_df.columns:
        vol_mass_df = vol_mass_df.rename(columns={"frame": "Frame"})

    if "Frame" not in voxel_mass_df.columns or "Frame" not in vol_mass_df.columns:
        raise ValueError("Both CSVs must have a frame column (frame or Frame).")

    # Merge dataframes on 'name' and 'Frame'
    merged_df = pd.merge(
        voxel_mass_df,
        vol_mass_df,
        on=["name", "Frame"],
        suffixes=("_voxel", "_vol"),
    )
    merged_df.rename(columns={'Mass_voxel': 'Voxel_Mass', 'Mass_vol': 'Vol_Mass'}, inplace=True)
    # Save merged data to CSV

    merged_df.to_csv(output_csv, float_format="%.3f", index=False)
    print(f"✓ Merged mass data saved: {output_csv}")
    return merged_df

def _plot_analysis_suite(vol, voxel, output_path, title_prefix):
    diffs = vol - voxel
    valid_mask = vol.notna() & voxel.notna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # QQ plot of differences
    stats.probplot(diffs.dropna(), dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title(f"{title_prefix} - QQ Plot (Vol - Voxel)")

    # EOV (residuals vs fitted/vol)
    axes[0, 1].scatter(vol[valid_mask], diffs[valid_mask], alpha=0.6)
    axes[0, 1].axhline(0, color="gray", linewidth=1)
    axes[0, 1].set_title(f"{title_prefix} - EOV: Residuals vs Vol")
    axes[0, 1].set_xlabel("Vol Mass")
    axes[0, 1].set_ylabel("Vol - Voxel")
    axes[0, 1].grid(True, alpha=0.3)

    # Cook's distance (requires statsmodels)
    try:
        X = sm.add_constant(vol[valid_mask])
        y = voxel[valid_mask]
        model = sm.OLS(y, X).fit()
        influence = model.get_influence()
        cooks = influence.cooks_distance[0]
        axes[1, 0].stem(range(len(cooks)), cooks, basefmt=" ")
        axes[1, 0].axhline(4 / len(cooks), color="red", linestyle="--", linewidth=1)
        axes[1, 0].set_title(f"{title_prefix} - Cook's Distance")
        axes[1, 0].set_xlabel("Observation")
        axes[1, 0].set_ylabel("Cook's D")
    except Exception:
        axes[1, 0].text(
            0.5,
            0.5,
            "Cook's Distance requires statsmodels",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_axis_off()

    # Box and whisker of differences
    axes[1, 1].boxplot(diffs.dropna(), vert=True)
    axes[1, 1].axhline(0, color="gray", linewidth=1)
    axes[1, 1].set_title(f"{title_prefix} - Box & Whisker (Vol - Voxel)")
    axes[1, 1].set_ylabel("Mass difference")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_case_comparison(merged_df, output_dir):
    if "name" not in merged_df.columns:
        raise ValueError("merged_df must contain a 'name' column for case plots")

    for case_name, case_df in merged_df.groupby("name", dropna=False):
        if pd.isna(case_name):
            continue

        case_dir = Path(output_dir) / "cases" / str(case_name)
        case_dir.mkdir(parents=True, exist_ok=True)

        if {"lvm_voxel", "lvm_vol", "rvm_voxel", "rvm_vol"}.issubset(case_df.columns):
            case_df = case_df.sort_values("Frame")
            frame_min = 0
            total_frames = case_df["Frame"].nunique()
            frame_max = max(total_frames - 1, 0)

            lvm_vals = pd.concat([case_df["lvm_vol"], case_df["lvm_voxel"]]).dropna()
            rvm_vals = pd.concat([case_df["rvm_vol"], case_df["rvm_voxel"]]).dropna()
            lvm_pad = (lvm_vals.max() - lvm_vals.min()) * 0.08 if not lvm_vals.empty else 0
            rvm_pad = (rvm_vals.max() - rvm_vals.min()) * 0.08 if not rvm_vals.empty else 0

            lvm_diff = case_df["lvm_vol"] - case_df["lvm_voxel"]
            rvm_diff = case_df["rvm_vol"] - case_df["rvm_voxel"]

            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)

            axes[0, 0].plot(case_df["Frame"], case_df["lvm_vol"], marker="o", label="LVM vol")
            axes[0, 0].plot(case_df["Frame"], case_df["lvm_voxel"], marker="o", label="LVM voxel")
            axes[0, 0].set_title(f"{case_name} - LVM")
            axes[0, 0].set_ylabel("Mass")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlim(frame_min, frame_max)
            if not lvm_vals.empty:
                axes[0, 0].set_ylim(lvm_vals.min() - lvm_pad, lvm_vals.max() + lvm_pad)

            axes[0, 1].plot(case_df["Frame"], case_df["rvm_vol"], marker="o", label="RVM vol")
            axes[0, 1].plot(case_df["Frame"], case_df["rvm_voxel"], marker="o", label="RVM voxel")
            axes[0, 1].set_title(f"{case_name} - RVM")
            axes[0, 1].set_ylabel("Mass")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim(frame_min, frame_max)
            if not rvm_vals.empty:
                axes[0, 1].set_ylim(rvm_vals.min() - rvm_pad, rvm_vals.max() + rvm_pad)

            axes[1, 0].bar(case_df["Frame"], lvm_diff, alpha=0.5, width=0.6)
            axes[1, 0].axhline(0, color="gray", linewidth=1)
            axes[1, 0].set_title(f"{case_name} - LVM diff (vol - voxel)")
            axes[1, 0].set_xlabel("Frame")
            axes[1, 0].set_ylabel("Mass difference")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim(frame_min, frame_max)

            axes[1, 1].bar(case_df["Frame"], rvm_diff, alpha=0.5, width=0.6)
            axes[1, 1].axhline(0, color="gray", linewidth=1)
            axes[1, 1].set_title(f"{case_name} - RVM diff (vol - voxel)")
            axes[1, 1].set_xlabel("Frame")
            axes[1, 1].set_ylabel("Mass difference")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim(frame_min, frame_max)

            output_path = case_dir / f"{case_name}_mass_compare.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)


def mass_data_analysis(merged_df, output_dir): 
    """
    Docstring for mass_data_analysis
    TODO: 
    1. QQ plot 
    2. EOV
    3. Cook's Distance
    4. Box and whisker 
    """

    if "name" not in merged_df.columns:
        raise ValueError("merged_df must contain a 'name' column for case analysis")

    for case_name, case_df in merged_df.groupby("name", dropna=False):
        if pd.isna(case_name):
            continue

        case_dir = Path(output_dir) / "cases" / str(case_name)
        case_dir.mkdir(parents=True, exist_ok=True)

        if {"Voxel_Mass", "Vol_Mass"}.issubset(case_df.columns):
            voxel = case_df["Voxel_Mass"]
            vol = case_df["Vol_Mass"]
            output_path = case_dir / f"{case_name}_mass_data_analysis.png"
            _plot_analysis_suite(vol, voxel, output_path, str(case_name))
            continue
        elif {"lvm_voxel", "lvm_vol", "rvm_voxel", "rvm_vol"}.issubset(case_df.columns):
            lv_voxel = case_df["lvm_voxel"]
            lv_vol = case_df["lvm_vol"]
            rv_voxel = case_df["rvm_voxel"]
            rv_vol = case_df["rvm_vol"]

            lv_output = case_dir / f"{case_name}_lv_mass_data_analysis.png"
            rv_output = case_dir / f"{case_name}_rv_mass_data_analysis.png"
            _plot_analysis_suite(lv_vol, lv_voxel, lv_output, f"{case_name} - LV")
            _plot_analysis_suite(rv_vol, rv_voxel, rv_output, f"{case_name} - RV")
            continue
        else:
            raise ValueError("Required mass columns not found in merged_df.")

    


if __name__ == "__main__":
    # ### Test files 
    # volMass_csv = r"C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_compare\test_volMass.csv"
    # voxelMass_csv = r"C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_compare\test_voxelMass.csv"

    voxelMass_csv = r"Z:\sandboxes\Jerry\hpc biv-me data\analysis\Voxel_Mass\Tof_44cases_thickness_thickness_analysis_20260202_161010\mass_summary.csv"
    volMass_csv = r"C:\Users\jchu579\Documents\SRS 2025_26\bivme-data\analysis\Vol_Time_plots_before_troubleshoot\lvrv_volumes.csv"
    output_dir = Path(r"C:\Users\jchu579\Documents\SRS 2025_26\dev_space\output_compare")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = str(plots_dir / f"mass_compare_detail_{timestamp}.csv")
    summary_csv = str(plots_dir / f"mass_compare_summary_{timestamp}.csv")

    merged_df = merge_mass_data(output_csv, voxelMass_csv, volMass_csv)
    plot_case_comparison(merged_df, output_dir)
    mass_data_analysis(merged_df, output_dir)


