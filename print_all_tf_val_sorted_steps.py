from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE = Path('/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517')
COMPARISON = BASE / 'PT_TF_NONE_TRAIN_VAL_COMPARISON_20260523.tsv'


def read_tsv(path: Path) -> List[Dict[str, str]]:
	with path.open(newline='') as handle:
		return list(csv.DictReader(handle, delimiter='\t'))


def fmt(value: str, digits: int = 4) -> str:
	try:
		return f'{float(value):.{digits}g}'
	except Exception:
		return value


def case_base(case_id: str) -> str:
	for suffix in ('__best_total', '__final'):
		if case_id.endswith(suffix):
			return case_id[:-len(suffix)]
	return case_id


def load_train_meta() -> Dict[str, Dict[str, str]]:
	meta: Dict[str, Dict[str, str]] = {}
	for name in ('TRAINING_SUMMARY.tsv', 'TRAINING_SUMMARY_10x_COMPLETED.tsv'):
		path = BASE / name
		if not path.exists():
			continue
		for row in read_tsv(path):
			meta[row['case_id']] = row
	if COMPARISON.exists():
		for row in read_tsv(COMPARISON):
			if row['framework'] == 'TF':
				base = case_base(row['case_id'])
				meta.setdefault(base, {})
				meta[base]['run_dir'] = str(BASE / 'runs' / 'none_100k' / base)
				meta[base]['wall_time_s'] = row.get('TRAIN_runtime_s', '')
				meta[base]['steps'] = row.get('steps', '')
	return meta


def lcurve_row(run_dir: Path, target_step: int) -> Tuple[int, float, float, float]:
	path = run_dir / 'lcurve.out'
	best: Optional[Tuple[int, float, float, float]] = None
	best_dist: Optional[int] = None
	with path.open() as handle:
		for line in handle:
			line = line.strip()
			if not line or line.startswith('#'):
				continue
			parts = line.split()
			if len(parts) < 4:
				continue
			step = int(float(parts[0]))
			dist = abs(step - target_step)
			if best_dist is None or dist < best_dist:
				best_dist = dist
				best = (step, float(parts[1]), float(parts[2]), float(parts[3]))
			if dist == 0:
				break
	if best is None:
		raise ValueError(f'No lcurve row found in {path}')
	return best


def lcurve_best_total_step(run_dir: Path) -> Tuple[int, float]:
	path = run_dir / 'lcurve.out'
	best_step: Optional[int] = None
	best_total: Optional[float] = None
	with path.open() as handle:
		for line in handle:
			line = line.strip()
			if not line or line.startswith('#'):
				continue
			parts = line.split()
			if len(parts) < 2:
				continue
			step = int(float(parts[0]))
			total = float(parts[1])
			if best_total is None or total < best_total:
				best_total = total
				best_step = step
	if best_step is None or best_total is None:
		raise ValueError(f'No lcurve rows found in {path}')
	return best_step, best_total


def gather_validated_tf_rows() -> List[Dict[str, str]]:
	train_meta = load_train_meta()
	rows: Dict[str, Dict[str, str]] = {}
	for summary in BASE.glob('pseudo_validation*/PSEUDO_VALIDATION_SUMMARY.tsv'):
		for row in read_tsv(summary):
			if row.get('row_type') != 'aggregate' or row.get('dataset_id') != 'all' or row.get('split') != 'all':
				continue
			case_id = row['case_id']
			base = case_base(case_id)
			meta = train_meta.get(base)
			if not meta:
				continue
			run_dir = Path(meta['run_dir'])
			total_steps = int(float(meta.get('final_step') or meta.get('steps') or row.get('steps') or 0))
			actual_step, train_total, train_e, train_f = lcurve_row(run_dir, total_steps)
			best_step, _ = lcurve_best_total_step(run_dir)
			rows[case_id] = {
				'name': case_id,
				'checkpoint_step': str(actual_step),
				'total_steps': str(total_steps),
				'best_total_loss_step': str(best_step),
				'TRAIN_total_loss': f'{train_total:.8g}',
				'TRAIN_E': f'{train_e:.8g}',
				'VAL_E': row['energy_rmse_per_atom'],
				'TRAIN_F': f'{train_f:.8g}',
				'VAL_F': row['force_rmse'],
			}

	if COMPARISON.exists():
		for row in read_tsv(COMPARISON):
			if row['framework'] != 'TF':
				continue
			base = case_base(row['case_id'])
			best_step = row['checkpoint_step']
			run_dir = train_meta.get(base, {}).get('run_dir', '')
			if run_dir and Path(run_dir).exists():
				try:
					best_step = str(lcurve_best_total_step(Path(run_dir))[0])
				except Exception:
					best_step = row['checkpoint_step']
			rows[row['case_id']] = {
				'name': row['case_id'],
				'checkpoint_step': row['checkpoint_step'],
				'total_steps': row['steps'],
				'best_total_loss_step': best_step,
				'TRAIN_total_loss': row['TRAIN_total_loss_at_checkpoint'],
				'TRAIN_E': row['TRAIN_E_RMSE_per_atom_eV'],
				'VAL_E': row['VAL_E_RMSE_per_atom_eV'],
				'TRAIN_F': row['TRAIN_F_RMSE_eV_per_A'],
				'VAL_F': row['VAL_F_RMSE_eV_per_A'],
			}
	return sorted(rows.values(), key=lambda r: float(r['VAL_E']))


def main() -> None:
	rows = gather_validated_tf_rows()
	print('| name | checkpoint step | final/total steps | best TRAIN total-loss step | TRAIN total loss | TRAIN E RMSE (eV/atom) | VAL E RMSE (eV/atom) | TRAIN F RMSE (eV/A) | VAL F RMSE (eV/A) |')
	print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
	for row in rows:
		print(f"| {row['name']} | {row['checkpoint_step']} | {row['total_steps']} | {row['best_total_loss_step']} | {fmt(row['TRAIN_total_loss'])} | {fmt(row['TRAIN_E'])} | {fmt(row['VAL_E'])} | {fmt(row['TRAIN_F'])} | {fmt(row['VAL_F'])} |")


if __name__ == '__main__':
	main()
