import subprocess
import os
import shutil
import time
from pathlib import Path

ROUNDS = 12
BASE_DIR = Path("/Data2/ds_24901720/Nadeem/fedquant_llama")
RESULTS_DIR = BASE_DIR / "results"
ROUNDS_DIR = RESULTS_DIR / "rounds"
ROUNDS_DIR.mkdir(parents=True, exist_ok=True)

total_start = time.time()
round_times = []

print("=" * 60)
print("FEDQUANT — 20 Round Federated Training")
print("=" * 60)

for round_num in range(1, ROUNDS + 1):
    round_start = time.time()

    print(f"\n{'='*60}")
    print(f"ROUND {round_num} / {ROUNDS}")
    print(f"{'='*60}")

    prev_adapter = "" if round_num == 1 else str(RESULTS_DIR / "aggregated")

    # Run each client
    for client in ["client2_sim_6gb", "client3_sim_6gb",
                   "client4_sim_8gb", "client5_sim_12gb"]:
        client_start = time.time()
        print(f"\n[Round {round_num}] Running {client}...")
        env = os.environ.copy()
        env["FEDQUANT_ROUND"] = str(round_num)
        env["FEDQUANT_PREV_ADAPTER"] = prev_adapter
        subprocess.run(["python3", f"clients/{client}.py"], cwd=BASE_DIR, env=env)
        client_time = time.time() - client_start
        print(f" {client} took: {client_time/60:.1f} minutes")

    # Run aggregation server
    print(f"\n[Round {round_num}] Running aggregation server...")
    subprocess.run(["python3", "server/aggregator.py"], cwd=BASE_DIR)

    # Save round snapshot
    round_dir = ROUNDS_DIR / f"round_{round_num:02d}"
    round_dir.mkdir(exist_ok=True)
    if (RESULTS_DIR / "aggregated").exists():
        shutil.copytree(RESULTS_DIR / "aggregated", round_dir / "aggregated", dirs_exist_ok=True)

    round_time = time.time() - round_start
    round_times.append(round_time)
    print(f"\n Round {round_num} complete!")
    print(f"  Round time: {round_time/60:.1f} minutes")
    elapsed = time.time() - total_start
    remaining = (round_time * (ROUNDS - round_num))
    print(f"  Total elapsed: {elapsed/3600:.2f} hours")
    print(f"  Estimated remaining: {remaining/3600:.2f} hours")

total_time = time.time() - total_start
print("\n" + "=" * 60)
print("ALL 12 ROUNDS COMPLETE")
print(f"Total training time: {total_time/3600:.2f} hours")
print(f"Average time per round: {(total_time/ROUNDS)/60:.1f} minutes")
print("=" * 60)
