import subprocess
import time
from pathlib import Path

configs = [
    "config_abl_set1b_style1e9.json",  # Set 1 Variant B (3 pairs)
    "config_abl_set1c_content10.json",  # Set 1 Variant C (3 pairs)
    "config_abl_set2a_shallow.json",  # Set 2 Shallow (3 pairs)
    "config_abl_set2b_deep.json",  # Set 2 Deep (3 pairs)
    "config_abl_set2c_multi.json",  # Set 2 Multi-layer (3 pairs)
    "config_abl_set3a_lr001.json",  # Set 3 LR=0.01 (2 pairs)
    "config_abl_set3b_lr01.json",  # Set 3 LR=0.1 (2 pairs)
]

print("=" * 80)
print("RUNNING ABLATION STUDY EXPERIMENTS")
print("=" * 80)
print(f"\nTotal configs: {len(configs)}")
print("Estimated time: 1-2 hours")
print("=" * 80)

for idx, config in enumerate(configs, 1):
    print(f"\n[{idx}/{len(configs)}] Running: {config}")
    print("-" * 80)

    if not Path(config).exists():
        print(f" Config not found: {config}")
        continue

    start_time = time.time()

    # Run experiment
    result = subprocess.run(
        ["python", "experiment_runner.py", "ablation", config], capture_output=False
    )

    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"Completed in {elapsed/60:.1f} minutes")
    else:
        print(f"Failed with code {result.returncode}")

    print("-" * 80)

print("\n" + "=" * 80)
print("ALL ABLATION EXPERIMENTS COMPLETE!")
print("=" * 80)
