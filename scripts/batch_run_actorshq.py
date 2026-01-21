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
ACTORS = ["Actor01", "Actor02", "Actor03", "Actor04", "Actor05", "Actor06", "Actor07", "Actor08"]
SEQUENCES = ["Sequence1"]  # e.g., ["Sequence1", "Sequence2"]
FRAME_IDS = [0]

# Method: "train" or "eval"
METHOD = "train"

# GPUs to use (list of GPU IDs, e.g., ["0", "1", "2", "3"] or ["0"])
# Jobs will be distributed across these GPUs in parallel
CUDA_DEVICES = ["0", "1"]

# Base data directory
BASE_DATA_DIR = "/synology/actorshq/colmap"

# Resolution (1, 2, or 4)
RESOLUTION = 4

# Root run path (working directory for running experiments)
ROOT_RUN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# ================================================


@dataclass
class RunConfig:
    actor: str
    sequence: str
    frame_id: int
    method: str = "train"
    cuda_device: str = "1"


def build_data_dir(actor: str, sequence: str, resolution: int = 4) -> str:
    """Build the data directory path for a given actor and sequence."""
    return f"{BASE_DATA_DIR}/{actor}/{sequence}/{resolution}x/frames"


def run_single_experiment(config: RunConfig):
    """Run a single experiment with the given configuration."""
    data_dir = build_data_dir(config.actor, config.sequence, RESOLUTION)

    print(f"\n{'='*60}")
    print(f"Running: Actor={config.actor}, Sequence={config.sequence}, Frame={config.frame_id}")
    print(f"Data dir: {data_dir}")
    print(f"Method: {config.method}")
    print(f"{'='*60}\n")

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = config.cuda_device

    # Build the command - we'll modify the config via command line or temp config
    # For simplicity, we'll create a modified version of the script inline
    script_content = f'''
import sys
import os
sys.path.insert(0, "{ROOT_RUN_PATH}")
from examples.simple_trainer import main2
from gsplat.strategy import DefaultStrategy
from examples.config import Config, load_config_from_toml, merge_config
from scripts.utils import set_result_dir

os.environ["CUDA_VISIBLE_DEVICES"] = "{config.cuda_device}"

# Build default config
default_cfg = Config(strategy=DefaultStrategy(verbose=True))
default_cfg.adjust_steps(default_cfg.steps_scaler)

# Load template config
template_path = "./configs/actorshq.toml"
cfg = load_config_from_toml(template_path)
cfg = merge_config(default_cfg, cfg)

# Override data directory
cfg.data_dir = "{data_dir}/frame{config.frame_id}"

# Build experiment name
exp_name = f"{config.actor}_{config.sequence}_l1_{{1.0 - cfg.ssim_lambda}}_ssim_{{cfg.ssim_lambda}}"
if cfg.masked_l1_loss:
    exp_name += f"_ml1_{{cfg.masked_l1_lambda}}"
if cfg.masked_ssim_loss:
    exp_name += f"_mssim_{{cfg.masked_ssim_lambda}}"
if cfg.alpha_loss:
    exp_name += f"_alpha_{{cfg.alpha_lambda}}"
if cfg.scale_var_loss:
    exp_name += f"_svar_{{cfg.scale_var_lambda}}"
if cfg.random_bkgd:
    exp_name += "_rbkgd"

cfg.disable_viewer = True
frame_id = {config.frame_id}

if "{config.method}" == "train":
    cfg.exp_name = exp_name
    cfg.scene_id = frame_id
    set_result_dir(cfg, exp_name)
    cfg.run_mode = "train"
    cfg.save_ply = True
    cfg.max_steps = 30000
    cfg.save_steps = list(sorted(set(range(0, cfg.max_steps + 1, 10000)) | {{1}}))
    cfg.ply_steps = cfg.save_steps
    cfg.eval_steps = cfg.save_steps
    cfg.init_type = "sfm"
    cfg.strategy = DefaultStrategy(verbose=True)

    print(f"Training frame {{frame_id}}")
    print(f"exp_name={{cfg.exp_name}}, scene_id={{cfg.scene_id}}, run_mode={{cfg.run_mode}}")
    main2(0, 0, 1, cfg)

elif "{config.method}" == "eval":
    cfg.exp_name = exp_name
    cfg.run_mode = "eval"
    cfg.init_type = "sfm"
    cfg.save_ply = False
    cfg.scene_id = frame_id
    set_result_dir(cfg, exp_name=exp_name)
    iter = cfg.max_steps
    ckpt = os.path.join(f"{{cfg.result_dir}}/ckpts/ckpt_{{iter - 1}}_rank0.pt")
    cfg.ckpt = ckpt

    print(f"Evaluating frame {{frame_id}}")
    main2(0, 0, 1, cfg)
'''

    # Write temp script and run it
    temp_script = f"/tmp/run_actorshq_{config.actor}_{config.sequence}_{config.frame_id}.py"
    with open(temp_script, "w") as f:
        f.write(script_content)

    # Run the script
    cmd = ["python", temp_script]
    result = subprocess.run(cmd, env=env, cwd=ROOT_RUN_PATH)

    # Clean up temp script
    os.remove(temp_script)

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
