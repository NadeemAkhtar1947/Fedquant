from huggingface_hub import snapshot_download

print("Downloading LLaMA 3.1 8B Instruct...")
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="./models/llama-3.1-8b",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"]
)
print("LLaMA 3.1 8B downloaded!")
