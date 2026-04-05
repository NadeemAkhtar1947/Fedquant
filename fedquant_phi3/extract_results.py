import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LOG_FILE = "output_20rounds_$PBS_JOBID.log"
OUT_DIR  = Path("results/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Reading log file...")
lines = open(LOG_FILE).readlines()
print(f"  Total lines: {len(lines)}")

clients = ["client2_sim_6gb", "client3_sim_6gb",
           "client4_sim_8gb", "client5_sim_12gb"]
client_markers = {
    "CLIENT 2": "client2_sim_6gb",
    "CLIENT 3": "client3_sim_6gb",
    "CLIENT 4": "client4_sim_8gb",
    "CLIENT 5": "client5_sim_12gb",
}
client_labels = {
    "client2_sim_6gb": "Client 2 (r=8,  6GB)",
    "client3_sim_6gb": "Client 3 (r=8,  6GB)",
    "client4_sim_8gb": "Client 4 (r=16, 8GB)",
    "client5_sim_12gb":"Client 5 (r=32, 12GB)",
}
colors = {
    "client2_sim_6gb": "#2196F3",
    "client3_sim_6gb": "#4CAF50",
    "client4_sim_8gb": "#FF9800",
    "client5_sim_12gb":"#E91E63",
}

# Parse using CLIENT marker + round detection inside block
all_data = {c: {} for c in clients}
current_client = None
current_round  = None
current_losses = []

for line in lines:
    # Detect client block start
    for marker, client in client_markers.items():
        if marker + " " in line and "Simulated" in line:
            # Save previous block
            if current_client and current_round and current_losses:
                if current_round not in all_data[current_client]:
                    all_data[current_client][current_round] = []
                all_data[current_client][current_round].extend(current_losses)
            current_client = client
            current_round  = None
            current_losses = []
            break

    # Detect round from "Loaded adapter from round: N"
    m = re.search(r'Loaded adapter from round:\s+(\d+)', line)
    if m and current_client:
        current_round = int(m.group(1))
        continue
    # Detect round from "Loaded + padded adapter to rN" (client4/5)
    m = re.search(r'Loaded \+ padded adapter to r=\d+', line)
    if m and current_client and current_round is None:
        # Count how many times this client has been seen to get round number
        seen = len(all_data[current_client])
        current_round = seen + 1
        continue

    # Round 1 marker → starting from base model
    if "Round 1 — starting from base model" in line and current_client:
        current_round = 1
        continue
    if "Round 1 - starting from base model" in line and current_client:
        current_round = 1
        continue

    # Detect loss
    m = re.search(r'Loss:\s+([\d.]+)', line)
    if m and current_client and current_round:
        current_losses.append(float(m.group(1)))

# Save last block
if current_client and current_round and current_losses:
    if current_round not in all_data[current_client]:
        all_data[current_client][current_round] = []
    all_data[current_client][current_round].extend(current_losses)

# Report
print("Rounds found per client:")
for c in clients:
    rounds = sorted(all_data[c].keys())
    if rounds:
        final = np.mean(all_data[c][rounds[-1]])
        print(f"  {c}: {len(rounds)} rounds {rounds}, final loss = {final:.4f}")
    else:
        print(f"  {c}: 0 rounds")

# Compute avg loss per round
rounds_list = list(range(1, 13))
avg_loss = {c: [] for c in clients}
for c in clients:
    for r in rounds_list:
        losses = all_data[c].get(r, [])
        avg_loss[c].append(np.mean(losses) if losses else None)

# Fill missing round 20 for any client using last known value
for c in clients:
    rounds = sorted(all_data[c].keys())
    if rounds and max(rounds) < 20:
        all_data[c][20] = all_data[c][max(rounds)]
        print(f"  Filled round 20 for {c} using round {max(rounds)}")

# Save JSON
with open(OUT_DIR / "loss_data.json", "w") as f:
    json.dump({"rounds": rounds_list,
               "avg_loss_per_round": {c: avg_loss[c] for c in clients}}, f, indent=2)
print("Loss data saved")

# Plot 1 - Loss curves
fig, ax = plt.subplots(figsize=(12, 6))
for c in clients:
    valid_rounds = [r for r, v in zip(rounds_list, avg_loss[c]) if v is not None]
    valid_vals   = [v for v in avg_loss[c] if v is not None]
    if valid_vals:
        ax.plot(valid_rounds, valid_vals, marker='o', markersize=4,
                label=client_labels[c], color=colors[c], linewidth=2)
ax.set_xlabel("Federated Learning Round", fontsize=13)
ax.set_ylabel("Average Training Loss", fontsize=13)
ax.set_title("FedQuant — Loss Convergence Across 12 FL Rounds (Phi-3-mini)", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(rounds_list)
plt.tight_layout()
plt.savefig(OUT_DIR / "loss_curves.png", dpi=150, bbox_inches='tight')
print("Plot saved: loss_curves.png")

# Plot 2 - Final loss bar chart
fig, ax = plt.subplots(figsize=(8, 5))
final_losses, labels, bar_colors = [], [], []
for c in clients:
    vals = [x for x in avg_loss[c] if x is not None]
    if vals:
        final_losses.append(vals[-1])
        labels.append(client_labels[c])
        bar_colors.append(colors[c])
if final_losses:
    bars = ax.bar(labels, final_losses, color=bar_colors, edgecolor='white', width=0.5)
    ax.bar_label(bars, fmt='%.3f', fontsize=11, padding=3)
    ax.set_ylabel("Final Loss (Round 12)", fontsize=13)
    ax.set_title("FedQuant — Final Loss by Client Tier", fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(final_losses) * 1.3)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "final_loss_comparison.png", dpi=150, bbox_inches='tight')
    print("Plot saved: final_loss_comparison.png")

# Plot 3 - Timing
timing_data = {c: [] for c in ["client2","client3","client4","client5"]}
for line in lines:
    for c in ["client2","client3","client4","client5"]:
        m = re.search(rf'{c}_sim_\w+ took: ([\d.]+) minutes', line)
        if m:
            timing_data[c].append(float(m.group(1)))

fig, ax = plt.subplots(figsize=(12, 5))
# Limit to 12 rounds
for c in ["client2","client3","client4","client5"]:
    timing_data[c] = timing_data[c][:12]
for c, col, lbl in zip(
    ["client2","client3","client4","client5"],
    ["#2196F3","#4CAF50","#FF9800","#E91E63"],
    ["Client 2 (r=8)","Client 3 (r=8)","Client 4 (r=16)","Client 5 (r=32)"]
):
    times = timing_data[c]
    if times:
        ax.plot(range(1, len(times)+1), times, marker='o',
                markersize=4, label=lbl, color=col, linewidth=2)
ax.set_xlabel("FL Round", fontsize=13)
ax.set_ylabel("Training Time (minutes)", fontsize=13)
ax.set_title("FedQuant — Training Time per Client per Round", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "timing_curves.png", dpi=150, bbox_inches='tight')
print("Plot saved: timing_curves.png")

print("\nAll analysis complete!")
