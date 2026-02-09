import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from automate_thickness_processing import NIfTIAnalyzer, locate_thickness_files


def setup_argparse():
	parser = argparse.ArgumentParser(
		description=(
			"Find thickness outliers beyond a global percentile and plot max thickness "
			"after removing those outliers."
		)
	)
	parser.add_argument(
		"-mdir",
		"--model_dir",
		dest="model_dir",
		required=True,
		help="Model directory containing thickness NIfTI files",
	)
	parser.add_argument(
		"-o",
		"--output_dir",
		dest="output_dir",
		required=True,
		help="Output directory for CSV and plots",
	)
	parser.add_argument(
		"-p",
		"--percentile",
		dest="percentile",
		type=float,
		default=99.0,
		help="Percentile threshold to flag outliers (default: 99)",
	)
	parser.add_argument(
		"--workers",
		dest="workers",
		type=int,
		default=1,
		help="Number of worker processes to use (default: 1)",
	)
	parser.add_argument(
		"--task-list",
		dest="task_list",
		default=None,
		help="Path to case list file for job arrays",
	)
	parser.add_argument(
		"--task-index",
		dest="task_index",
		type=int,
		default=None,
		help="1-based index into task list (used by job arrays)",
	)
	parser.add_argument(
		"--write-task-list",
		dest="write_task_list",
		default=None,
		help="Write case list to file and exit",
	)
	parser.add_argument(
		"--threshold-file",
		dest="threshold_file",
		default=None,
		help="Path to JSON file storing the global percentile threshold",
	)
	parser.add_argument(
		"--write-threshold",
		dest="write_threshold",
		action="store_true",
		help="Compute global threshold and write it to --threshold-file",
	)
	parser.add_argument(
		"--merge-only",
		dest="merge_only",
		action="store_true",
		help="Merge per-case outputs under output_dir and exit",
	)
	return parser.parse_args()


def _collect_global_values(thickness_files):
	all_values = []
	for file_path in thickness_files.keys():
		analyzer = NIfTIAnalyzer(file_path)
		valid_data = analyzer.get_valid_data()
		if valid_data.size:
			all_values.append(valid_data)

	if not all_values:
		return np.array([])

	return np.concatenate(all_values)


def _group_files_by_case(thickness_files):
	cases = {}
	for file_path, (thickness_type, folder_name, frame) in thickness_files.items():
		cases.setdefault(folder_name, []).append((file_path, thickness_type, frame))
	return cases


def _write_case_list(cases, output_path):
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		for case in sorted(cases):
			handle.write(f"{case}\n")
	return output_path


def _read_case_list(case_list_path):
	case_list_path = Path(case_list_path)
	if not case_list_path.exists():
		raise FileNotFoundError(f"Case list not found: {case_list_path}")
	with case_list_path.open("r", encoding="utf-8") as handle:
		cases = [line.strip() for line in handle if line.strip()]
	if not cases:
		raise ValueError("Case list is empty.")
	return cases


def _write_threshold(threshold_file, percentile, threshold, count):
	payload = {
		"percentile": percentile,
		"threshold": threshold,
		"count": int(count),
		"created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
	}
	threshold_file = Path(threshold_file)
	threshold_file.parent.mkdir(parents=True, exist_ok=True)
	threshold_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	return threshold_file


def _read_threshold(threshold_file):
	threshold_file = Path(threshold_file)
	if not threshold_file.exists():
		raise FileNotFoundError(f"Threshold file not found: {threshold_file}")
	payload = json.loads(threshold_file.read_text(encoding="utf-8"))
	if "threshold" not in payload:
		raise ValueError("Threshold file missing 'threshold' value.")
	return float(payload["threshold"])


def _find_outliers_for_file(file_path, threshold, folder_name, frame, thickness_type):
	analyzer = NIfTIAnalyzer(file_path)
	data = analyzer.get_fdata()
	valid_mask = (data > 0) & ~np.isnan(data)
	outlier_mask = valid_mask & (data > threshold)

	if not np.any(outlier_mask):
		return []

	outlier_indices = np.argwhere(outlier_mask)
	outliers = []
	for y, x, slice_idx in outlier_indices:
		outliers.append(
			{
				"folder_name": folder_name,
				"frame": frame,
				"thickness_type": thickness_type,
				"value": float(data[y, x, slice_idx]),
				"x": int(x),
				"y": int(y),
				"slice": int(slice_idx),
			}
		)

	return outliers


def _max_thickness_without_outliers(file_path, threshold):
	analyzer = NIfTIAnalyzer(file_path)
	data = analyzer.get_fdata()
	valid_mask = (data > 0) & ~np.isnan(data)
	filtered_values = data[valid_mask & (data <= threshold)]

	if filtered_values.size == 0:
		return np.nan

	return float(np.nanmax(filtered_values))


def _plot_max_thickness(max_df, output_path, title):
	fig, ax = plt.subplots(figsize=(14, 6))
	frame_num = pd.to_numeric(max_df["frame"], errors="coerce")
	max_df = max_df.assign(frame_num=frame_num)

	if max_df["frame_num"].notna().any():
		for thickness_type, group in max_df.groupby("thickness_type"):
			group = group.sort_values("frame_num")
			ax.plot(
				group["frame_num"],
				group["max_thickness"],
				marker="o",
				linewidth=1,
				label=thickness_type,
			)
		ax.set_xlabel("Frame")
	else:
		ax.plot(max_df["index"], max_df["max_thickness"], marker="o", linewidth=1)
		ax.set_xlabel("File Index")

	ax.set_title(title)
	ax.set_ylabel("Max Thickness (mm)")
	if max_df["thickness_type"].nunique() > 1:
		ax.legend()
	ax.grid(True, alpha=0.3)

	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def _process_case(case_name, case_files, threshold, output_dir, timestamp):
	outlier_rows = []
	max_rows = []

	for index, (file_path, thickness_type, frame) in enumerate(case_files, start=1):
		outlier_rows.extend(
			_find_outliers_for_file(file_path, threshold, case_name, frame, thickness_type)
		)
		max_thickness = _max_thickness_without_outliers(file_path, threshold)
		max_rows.append(
			{
				"index": index,
				"folder_name": case_name,
				"frame": frame,
				"thickness_type": thickness_type,
				"file_path": file_path,
				"max_thickness": max_thickness,
			}
		)

	case_dir = Path(output_dir) / "cases" / str(case_name)
	case_dir.mkdir(parents=True, exist_ok=True)

	outlier_columns = [
		"folder_name",
		"frame",
		"thickness_type",
		"value",
		"x",
		"y",
		"slice",
		"file_path",
	]
	outliers_df = pd.DataFrame(outlier_rows, columns=outlier_columns)
	max_df = pd.DataFrame(max_rows)

	outlier_csv = case_dir / f"thickness_outliers_{case_name}_{timestamp}.csv"
	max_csv = case_dir / f"max_thickness_filtered_{case_name}_{timestamp}.csv"
	plot_path = case_dir / f"max_thickness_filtered_{case_name}_{timestamp}.png"

	outliers_df.to_csv(outlier_csv, index=False)
	max_df.to_csv(max_csv, index=False)
	_plot_max_thickness(max_df, plot_path, f"{case_name} - Max Thickness (Outliers Removed)")

	return outliers_df, max_df


def _process_case_worker(payload):
	return _process_case(**payload)


def _merge_case_outputs(output_dir):
	output_dir = Path(output_dir)
	outlier_files = sorted(output_dir.glob("cases/**/thickness_outliers_*.csv"))
	max_files = sorted(output_dir.glob("cases/**/max_thickness_filtered_*.csv"))

	if not outlier_files and not max_files:
		raise ValueError("No per-case CSV files found under output_dir/cases.")

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	merged_outliers = output_dir / f"thickness_outliers_merged_{timestamp}.csv"
	merged_max = output_dir / f"max_thickness_filtered_merged_{timestamp}.csv"
	merged_plot = output_dir / f"max_thickness_filtered_merged_{timestamp}.png"

	if outlier_files:
		outliers_df = pd.concat([pd.read_csv(path) for path in outlier_files], ignore_index=True)
		outliers_df.to_csv(merged_outliers, index=False)
		print(f"Merged outliers saved: {merged_outliers}")

	if max_files:
		max_df = pd.concat([pd.read_csv(path) for path in max_files], ignore_index=True)
		max_df.to_csv(merged_max, index=False)
		_plot_max_thickness(max_df, merged_plot, "Max Thickness (Outliers Removed)")
		print(f"Merged max summary saved: {merged_max}")
		print(f"Merged plot saved: {merged_plot}")


def run_outlier_removal(
	model_dir,
	output_dir,
	percentile,
	workers,
	task_list,
	task_index,
	write_task_list,
	threshold_file,
	write_threshold,
	merge_only,
):
	thickness_files = locate_thickness_files(model_dir)
	if not thickness_files:
		raise ValueError("No thickness files found in the model directory.")

	cases = _group_files_by_case(thickness_files)
	if write_task_list:
		output_path = _write_case_list(cases.keys(), write_task_list)
		print(f"Case list written: {output_path}")
		return

	if merge_only:
		_merge_case_outputs(output_dir)
		return

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	if threshold_file and Path(threshold_file).exists() and not write_threshold:
		threshold = _read_threshold(threshold_file)
	else:
		global_values = _collect_global_values(thickness_files)
		if global_values.size == 0:
			raise ValueError("No valid thickness values found across files.")
		threshold = np.percentile(global_values, percentile)
		if threshold_file:
			_write_threshold(threshold_file, percentile, float(threshold), len(global_values))
			print(f"Threshold written: {threshold_file}")

	print(f"Global {percentile:.2f}th percentile threshold: {threshold:.4f}")

	if task_list:
		cases_to_process = _read_case_list(task_list)
	else:
		cases_to_process = sorted(cases.keys())

	if task_index is not None:
		if task_index < 1 or task_index > len(cases_to_process):
			raise ValueError("task_index out of range for provided task list.")
		cases_to_process = [cases_to_process[task_index - 1]]

	payloads = []
	for case_name in cases_to_process:
		if case_name not in cases:
			continue
		payloads.append(
			{
				"case_name": case_name,
				"case_files": cases[case_name],
				"threshold": threshold,
				"output_dir": output_dir,
				"timestamp": timestamp,
			}
		)

	if workers > 1 and len(payloads) > 1:
		with ProcessPoolExecutor(max_workers=workers) as executor:
			list(executor.map(_process_case_worker, payloads))
	else:
		for payload in payloads:
			_process_case(**payload)


def main():
	args = setup_argparse()
	run_outlier_removal(
		args.model_dir,
		args.output_dir,
		args.percentile,
		args.workers,
		args.task_list,
		args.task_index,
		args.write_task_list,
		args.threshold_file,
		args.write_threshold,
		args.merge_only,
	)


if __name__ == "__main__":
	main()
