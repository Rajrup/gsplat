"""
Unified batch runner for gsplat experiments.
Supports multiple datasets: ActorsHQ, Neural 3D Video, etc.

Usage:
    python batch_run.py
"""
import subprocess
import os
import itertools
from dataclasses import dataclass
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


# ================= Configurations =================
DATASET = "neural3d" 
METHOD = "train"  # "train" or "eval"
CUDA_DEVICES = ["0", "1", "2", "3"]

DATASET_CONFIGS = {
    "actorshq": {
        "base_data_dir": "/synology/actorshq/colmap",
        "actors": ["Actor06", "Actor02"],
        "sequences": ["Sequence1"],
        "frame_ids": [1],
        "resolution": 4,
        "config_path": "./configs/actorshq.toml",
    },
    "neural3d": {
        "base_data_dir": "/synology/Neural_3D_Video",
        "sequences": ["coffee_martini", "cook_spinach", "cut_roasted_beef",
                      # "flame_salmon_1", "flame_steak", "sear_steak"
                      ],
        "frame_ids": [0],
        "config_path": "./configs/actorshq.toml",
    },
}
# =========================================================

@dataclass
class JobConfig:
    """Configuration for a single experiment job."""
    data_dir: str
    frame_id: int
    exp_name_prefix: str
    run_info: str
    method: str = "train"
    cuda_device: str = "0"
    config_path: str = "./configs/actorshq.toml"
    run_script_path: str = ""
    root_run_path: str = ""


def build_data_dir_actorshq(cfg: dict, actor: str, sequence: str, frame_id: int) -> str:
    """Build data directory path for ActorsHQ dataset."""
    resolution = cfg.get("resolution", 4)
    return f"{cfg['base_data_dir']}/{actor}/{sequence}/{resolution}x/frames/frame{frame_id}"


def build_data_dir_neural3d(cfg: dict, sequence: str, frame_id: int) -> str:
    """Build data directory path for Neural 3D Video dataset."""
    return f"{cfg['base_data_dir']}/{sequence}/colmap/frame{frame_id:05d}"


def create_jobs_actorshq(cfg: dict, method: str, cuda_devices: List[str],
                          run_script_path: str, root_run_path: str) -> List[JobConfig]:
    """Create job configs for ActorsHQ dataset."""
    actors = cfg.get("actors", [])
    sequences = cfg.get("sequences", [])
    frame_ids = cfg.get("frame_ids", [])
    config_path = cfg.get("config_path", "./configs/actorshq.toml")

    all_combinations = list(itertools.product(actors, sequences, frame_ids))
    num_gpus = len(cuda_devices)

    jobs = []
    for idx, (actor, sequence, frame_id) in enumerate(all_combinations):
        gpu_idx = idx % num_gpus
        exp_name_prefix = f"{actor}_{sequence}"
        data_dir = build_data_dir_actorshq(cfg, actor, sequence, frame_id)
        run_info = f"{actor}/{sequence}/frame{frame_id}"

        job = JobConfig(
            data_dir=data_dir,
            frame_id=frame_id,
            exp_name_prefix=exp_name_prefix,
            run_info=run_info,
            method=method,
            cuda_device=cuda_devices[gpu_idx],
            config_path=config_path,
            run_script_path=run_script_path,
            root_run_path=root_run_path,
        )
        jobs.append(job)

    return jobs


def create_jobs_neural3d(cfg: dict, method: str, cuda_devices: List[str],
                          run_script_path: str, root_run_path: str) -> List[JobConfig]:
    """Create job configs for Neural 3D Video dataset."""
    sequences = cfg.get("sequences", [])
    frame_ids = cfg.get("frame_ids", [])
    config_path = cfg.get("config_path", "./configs/actorshq.toml")

    all_combinations = list(itertools.product(sequences, frame_ids))
    num_gpus = len(cuda_devices)

    jobs = []
    for idx, (sequence, frame_id) in enumerate(all_combinations):
        gpu_idx = idx % num_gpus
        exp_name_prefix = f"neural3d_{sequence}"
        data_dir = build_data_dir_neural3d(cfg, sequence, frame_id)
        run_info = f"{sequence}/frame{frame_id:05d}"

        job = JobConfig(
            data_dir=data_dir,
            frame_id=frame_id,
            exp_name_prefix=exp_name_prefix,
            run_info=run_info,
            method=method,
            cuda_device=cuda_devices[gpu_idx],
            config_path=config_path,
            run_script_path=run_script_path,
            root_run_path=root_run_path,
        )
        jobs.append(job)

    return jobs


# Dataset name -> job creator function
JOB_CREATORS = {
    "actorshq": create_jobs_actorshq,
    "neural3d": create_jobs_neural3d,
}


def run_single_experiment(config: JobConfig) -> int:
    """Run a single experiment with the given configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {config.run_info}")
    print(f"Data dir: {config.data_dir}")
    print(f"Method: {config.method}")
    print(f"GPU: {config.cuda_device}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = config.cuda_device

    cmd = [
        "python", config.run_script_path,
        "--data_dir", config.data_dir,
        "--frame_id", str(config.frame_id),
        "--method", config.method,
        "--exp_name_prefix", config.exp_name_prefix,
        "--config", config.config_path,
        "--disable_viewer",
    ]

    result = subprocess.run(cmd, env=env, cwd=config.root_run_path)
    return result.returncode


def run_experiment_wrapper(args: Tuple[JobConfig, int, int]) -> Tuple[str, bool]:
    """Wrapper function for parallel execution."""
    config, job_idx, total_jobs = args
    print(f"[Job {job_idx + 1}/{total_jobs}] Starting on GPU {config.cuda_device}: {config.run_info}")

    return_code = run_single_experiment(config)

    if return_code == 0:
        print(f"[Job {job_idx + 1}/{total_jobs}] SUCCESS on GPU {config.cuda_device}: {config.run_info}")
        return (config.run_info, True)
    else:
        print(f"[Job {job_idx + 1}/{total_jobs}] FAILED on GPU {config.cuda_device}: {config.run_info}")
        return (config.run_info, False)


def run_batch(jobs: List[JobConfig], cuda_devices: List[str], dataset_name: str) -> None:
    """Run a batch of experiments across multiple GPUs."""
    print(f"Starting batch run for {dataset_name} experiments")

    total_experiments = len(jobs)
    num_gpus = len(cuda_devices)
    print(f"Total experiments: {total_experiments}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Running up to {num_gpus} experiments in parallel\n")

    indexed_jobs = [(job, idx, total_experiments) for idx, job in enumerate(jobs)]

    failed_runs = []
    successful_runs = []

    if num_gpus == 1:
        for job in indexed_jobs:
            run_info, success = run_experiment_wrapper(job)
            if success:
                successful_runs.append(run_info)
            else:
                failed_runs.append(run_info)
    else:
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(run_experiment_wrapper, job): job for job in indexed_jobs}
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


def main():
    cfg = DATASET_CONFIGS[DATASET]

    root_run_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_script_path = os.path.join(os.path.dirname(__file__), "run_actorshq.py")

    print(f"Dataset: {DATASET}")
    print(f"Method: {METHOD}")
    print(f"GPUs: {CUDA_DEVICES}")
    print(f"Config: {cfg}")

    # Create jobs using dataset-specific creator
    job_creator = JOB_CREATORS[DATASET]
    jobs = job_creator(cfg, METHOD, CUDA_DEVICES, run_script_path, root_run_path)

    run_batch(jobs, CUDA_DEVICES, DATASET)


if __name__ == "__main__":
    main()
