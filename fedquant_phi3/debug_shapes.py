# debug_shapes.py in your fedquant folder
from safetensors.torch import load_file

clients = {
    "client2": "./results/adapters/client2/adapter_model.safetensors",
    "client3": "./results/adapters/client3/adapter_model.safetensors",
    "client4": "./results/adapters/client4/adapter_model.safetensors",
    "client5": "./results/adapters/client5/adapter_model.safetensors",
}

# Print first 6 keys and shapes for each client
for client, path in clients.items():
    weights = load_file(path)
    print(f"\n{client}:")
    for i, (key, tensor) in enumerate(weights.items()):
        if i >= 6:
            break
        print(f"  {key}: {tensor.shape}")