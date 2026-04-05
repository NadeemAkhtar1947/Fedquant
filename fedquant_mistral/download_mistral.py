from huggingface_hub import snapshot_download

print("Downloading Mistral-7B-Instruct-v0.3...")
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir="./models/mistral-7b",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"]
)
print("Mistral-7B downloaded!")
