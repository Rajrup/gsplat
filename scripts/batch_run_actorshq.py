"""
Batch runner for ActorsHQ experiments.
Runs scripts/run_actorshq.py for different actor, sequence, and frame_id combinations.

Supports multiple GPUs for parallel execution. Jobs are automatically distributed
across available GPUs using a worker pool.
"""
import subprocess
import os
import itertools
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= Configuration =================
# Modify these lists to specify which actors, sequences, and frames to run
# ACTORS = ["Actor01", "Actor02", "Actor03", "Actor04", "Actor05", "Actor06", "Actor07", "Actor08"]
ACTORS = ["Actor06", "Actor02"]
SEQUENCES = ["Sequence1"]  # e.g., ["Sequence1", "Sequence2"]
FRAME_IDS = [1]

# Method: "train" or "eval"
METHOD = "train"

# GPUs to use (list of GPU IDs, e.g., ["0", "1", "2", "3"] or ["0"])
# Jobs will be distributed across these GPUs in parallel
CUDA_DEVICES = ["0", "1", "2", "3"]

# Base data directory
BASE_DATA_DIR = "/synology/actorshq/colmap"

# Resolution (1, 2, or 4)
RESOLUTION = 4

# Root run path (working directory for running experiments)
ROOT_RUN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to the run_actorshq.py script
RUN_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "run_actorshq.py")
# ================================================


@dataclass
class RunConfig:
    actor: str
    sequence: str
    frame_id: int
    method: str = "train"
    cuda_device: str = "1"


def build_data_dir(actor: str, sequence: str, frame_id: int, resolution: int = 4) -> str:
    """Build the data directory path for a given actor, sequence, and frame."""
    return f"{BASE_DATA_DIR}/{actor}/{sequence}/{resolution}x/frames/frame{frame_id}"


def run_single_experiment(config: RunConfig):
    """Run a single experiment with the given configuration."""
    data_dir = build_data_dir(config.actor, config.sequence, config.frame_id, RESOLUTION)
    exp_name_prefix = f"{config.actor}_{config.sequence}"

    print(f"\n{'='*60}")
    print(f"Running: Actor={config.actor}, Sequence={config.sequence}, Frame={config.frame_id}")
    print(f"Data dir: {data_dir}")
    print(f"Method: {config.method}")
    print(f"GPU: {config.cuda_device}")
    print(f"{'='*60}\n")

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = config.cuda_device

    # Build the command to call run_actorshq.py
    cmd = [
        "python", RUN_SCRIPT_PATH,
        "--data_dir", data_dir,
        "--frame_id", str(config.frame_id),
        "--method", config.method,
        "--exp_name_prefix", exp_name_prefix,
        "--disable_viewer",
    ]

    # Run the script
    result = subprocess.run(cmd, env=env, cwd=ROOT_RUN_PATH)

    return result.returncode


def run_experiment_wrapper(args):
    """Wrapper function for parallel execution."""
    config, job_idx, total_jobs = args
    run_info = f"{config.actor}/{config.sequence}/frame{config.frame_id}"
    print(f"[Job {job_idx + 1}/{total_jobs}] Starting on GPU {config.cuda_device}: {run_info}")

    return_code = run_single_experiment(config)

    if return_code == 0:
        print(f"[Job {job_idx + 1}/{total_jobs}] SUCCESS on GPU {config.cuda_device}: {run_info}")
        return (run_info, True)
    else:
        print(f"[Job {job_idx + 1}/{total_jobs}] FAILED on GPU {config.cuda_device}: {run_info}")
        return (run_info, False)


def main():
    """Run experiments for all combinations of actors, sequences, and frames.

    Jobs are distributed across available GPUs in parallel. Each GPU runs
    one job at a time, and new jobs are assigned to GPUs as they become free.
    """
    print(f"Starting batch run for ActorsHQ experiments")
    print(f"Actors: {ACTORS}")
    print(f"Sequences: {SEQUENCES}")
    print(f"Frame IDs: {FRAME_IDS}")
    print(f"Method: {METHOD}")
    print(f"GPUs: {CUDA_DEVICES}")

    total_experiments = len(ACTORS) * len(SEQUENCES) * len(FRAME_IDS)
    num_gpus = len(CUDA_DEVICES)
    print(f"Total experiments: {total_experiments}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Running up to {num_gpus} experiments in parallel\n")

    # Build all experiment configs, assigning GPUs in round-robin fashion
    all_combinations = list(itertools.product(ACTORS, SEQUENCES, FRAME_IDS))
    jobs = []
    for idx, (actor, sequence, frame_id) in enumerate(all_combinations):
        # Assign GPU in round-robin fashion for initial distribution
        # The ProcessPoolExecutor will handle the actual parallelism
        gpu_idx = idx % num_gpus
        config = RunConfig(
            actor=actor,
            sequence=sequence,
            frame_id=frame_id,
            method=METHOD,
            cuda_device=CUDA_DEVICES[gpu_idx],
        )
        jobs.append((config, idx, total_experiments))

    failed_runs = []
    successful_runs = []

    if num_gpus == 1:
        # Sequential execution for single GPU
        for job in jobs:
            run_info, success = run_experiment_wrapper(job)
            if success:
                successful_runs.append(run_info)
            else:
                failed_runs.append(run_info)
    else:
        # Parallel execution across multiple GPUs
        # Use ProcessPoolExecutor with max_workers = num_gpus
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(run_experiment_wrapper, job): job for job in jobs}

            for future in as_completed(futures):
                run_info, success = future.result()
                if success:
                    successful_runs.append(run_info)
                else:
                    failed_runs.append(run_info)

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH RUN SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(successful_runs) + len(failed_runs)}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Failed: {len(failed_runs)}")

    if failed_runs:
        print(f"\nFailed runs:")
        for run in failed_runs:
            print(f"  - {run}")


if __name__ == "__main__":
    main()
