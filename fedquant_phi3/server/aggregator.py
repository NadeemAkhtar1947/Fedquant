import torch
import os
import copy
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

print("FEDQUANT SERVER — Aggregation Server")

# ── Step 1: Check all adapters exist ──────────────
print("\n[Step 1] Checking client adapters...")

adapter_paths = {
    "client2": "./results/adapters/client2",
    "client3": "./results/adapters/client3",
    "client4": "./results/adapters/client4",
    "client5": "./results/adapters/client5",
}

for client, path in adapter_paths.items():
    if os.path.exists(path):
        print(f"{client}: found")
    else:
        print(f"{client}: MISSING — run that client script first")

# ── Step 2: Load each adapter into memory ─────────
print("\n[Step 2] Loading all adapters into memory...")

def load_adapter_weights(adapter_path):
    from safetensors.torch import load_file
    import json

    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    weights = load_file(weights_path)
    return weights

all_weights = {}
for client, path in adapter_paths.items():
    weights = load_adapter_weights(path)
    all_weights[client] = weights
    print(f"{client}: loaded {len(weights)} weight tensors")


# ── Step 3: Mixed Precision Aggregation ───────────
# Each client has different LoRA rank (r=8, r=8, r=16, r=32)
# We cannot average them directly — different shapes
# Solution: average only the layers that match in shape
print("\n[Step 3] Mixed-Precision Aggregation...")

client_weights = {
    "client2": 8  * 30000,
    "client3": 8  * 13000,
    "client4": 16 * 8000,
    "client5": 32 * 13999,
}
total = sum(client_weights.values())
client_weights = {k: v/total for k, v in client_weights.items()}
print("Client weights:")
for k, v in client_weights.items():
    print(f"  {k}: {v:.4f} ({v*100:.1f}%)")

TARGET_RANK = 8
print(f"Target rank: r={TARGET_RANK}")

all_keys = [set(w.keys()) for w in all_weights.values()]
common_keys = set.intersection(*all_keys)

aggregated = {}
averaged = 0

for key in common_keys:
    tensors_to_avg = []

    for client, weights in all_weights.items():
        tensor = weights[key].float()
        shape = tensor.shape

        # lora_A shape: (rank, hidden) → slice rows
        if shape[0] in [8, 16, 32] and shape[0] != TARGET_RANK:
            tensor = tensor[:TARGET_RANK, :]

        # lora_B shape: (hidden, rank) → slice columns
        elif shape[1] in [8, 16, 32] and shape[1] != TARGET_RANK:
            tensor = tensor[:, :TARGET_RANK]

        tensors_to_avg.append(tensor * client_weights[client])

    # Verify all shapes match now
    shapes = [t.shape for t in tensors_to_avg]
    if len(set(shapes)) == 1:
        aggregated[key] = sum(tensors_to_avg)
        averaged += 1
    else:
        aggregated[key] = all_weights["client2"][key].float()
        print(f"  Still mismatched: {key} shapes: {shapes}")

print(f"Layers successfully averaged: {averaged} out of {len(common_keys)}")


# ── Step 4: Save Aggregated Adapter ───────────────
print("\n[Step 4] Saving aggregated adapter...")

from safetensors.torch import save_file
import json
import shutil

os.makedirs("./results/aggregated", exist_ok=True)

# Save aggregated weights
save_file(aggregated, "./results/aggregated/adapter_model.safetensors")

# Copy adapter config from client2 (base config)
shutil.copy(
    "./results/adapters/client2/adapter_config.json",
    "./results/aggregated/adapter_config.json"
)

print("Saved to ./results/aggregated/")

# ── Step 5: Summary ────────────────────────────────
print("\n[Step 5] Round Summary...")
print(f"  Clients aggregated: {len(all_weights)}")
print(f"  Common layers averaged: {len(aggregated)}")
print(f"  Client weights used:")
for client, w in client_weights.items():
    print(f"    {client}: {w} ({int(w*100)}%)")

print("\n" + "=" * 50)
print("SERVER ROUND 1 COMPLETE")
print("Aggregated adapter saved to ./results/aggregated/")
print("This adapter can now be sent back to all clients")
