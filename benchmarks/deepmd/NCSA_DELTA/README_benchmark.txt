==============================================================================
  DeePMD-kit GPU Training Benchmark — NCSA Delta
  Date: 2026-04-04
  Author: akashgpt
==============================================================================

Overview
--------
Benchmarked DeePMD-kit v3.1.3 training on the three NVIDIA GPU types available
on NCSA Delta: A100, A40, and H200. The goal was to verify that the ALCHEMY
training pipeline (Apptainer container) works across all GPU partitions and to
quantify performance differences.

Setup
-----
- Software:    DeePMD-kit v3.1.3 (Apptainer image: deepmd-kit_latest.sif)
- Model:       se_e2_a descriptor, float64 precision
- System:      H-N (NH3-H2 mixture), 2 type_map elements
- Descriptor:  sel=[800,150], rcut=6.0, neuron=[25,50,100]
- Training:    8000 batches (numb_steps), batch_size=1 (auto)
- Data:        From ALCHEMY iteration v7_i35 (multiple zone/composition sets)
- Flag:        --skip-neighbor-stat used for benchmark only (skips ~30s startup)
- Account:     bguf-delta-gpu
- Container bind: /work/nvme

Results
-------
+---------------------+------------+-------------+------------------+-------------+----------+
| GPU                 | Partition  | GPU Memory  | Avg time/batch   | Wall time   | vs A100  |
+---------------------+------------+-------------+------------------+-------------+----------+
| NVIDIA H200         | gpuH200x8  | 141 GB      | 0.0118 s         |  101 s      | 2.0x     |
| NVIDIA A100-SXM4    | gpuA100x4  |  40 GB      | 0.0242 s         |  200 s      | 1.0x     |
| NVIDIA A40          | gpuA40x4   |  48 GB      | ~0.127 s         | ~1016 s *   | 0.19x    |
+---------------------+------------+-------------+------------------+-------------+----------+

* A40 projected from per-batch timing; job timed out at 10 min (reached batch 4100/8000).

Key Observations
----------------
1. All three GPU types work with the same Apptainer container and input file.
   No code changes, container rebuilds, or input modifications are needed.

2. H200 is ~2x faster than A100, and A100 is ~5.2x faster than A40.

3. The large A40 slowdown is due to FP64 (float64) precision in the model.
   FP64 throughput by GPU:
     - H200:  ~67 TFLOPS FP64
     - A100:  ~19.5 TFLOPS FP64
     - A40:   ~1.2 TFLOPS FP64
   The A40 is designed for visualization/FP32 workloads, not FP64 compute.
   If float32 precision were used, A40 performance would be much closer to A100.

4. For ALCHEMY production runs (which use float64), the recommended partition
   priority is:  H200 > A100 >> A40
   Use A40 only when H200/A100 queues are very long and wall time is not critical.

Files
-----
bench_A100.sh                 - SBATCH script for A100 benchmark
bench_A40.sh                  - SBATCH script for A40 benchmark
bench_H200.sh                 - SBATCH script for H200 benchmark
train_gpu_agnostic.apptr.sh   - GPU-agnostic template for production use
myinput.json                  - DeePMD input file used for benchmarking
slurm-A100-17256441.out       - Full A100 training log
slurm-A40-17256442.out        - Full A40 training log (timed out)
slurm-H200-17256443.out       - Full H200 training log

Reproducing
-----------
  cd $VASP_DATA/benchmarks/deepmd
  sbatch bench_A100.sh
  sbatch bench_A40.sh
  sbatch bench_H200.sh

Using the GPU-agnostic template:
  sbatch --partition=gpuA100x4 train_gpu_agnostic.apptr.sh myinput.json
  sbatch --partition=gpuH200x8 train_gpu_agnostic.apptr.sh myinput.json
  sbatch --partition=gpuA40x4  train_gpu_agnostic.apptr.sh myinput.json
