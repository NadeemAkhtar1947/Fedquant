from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from peft import PeftModel
import os

BASE_MODEL = "./models/phi3-mini"
ADAPTER    = "./results/rounds/round_20/aggregated"
OUTPUT     = "./merged_model"

print("Step 1: Loading base model...")
config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
if hasattr(config, 'rope_scaling') and config.rope_scaling:
    config.rope_scaling = {
        'type': 'longrope',
        'short_factor': [1.0] * 48,
        'long_factor': [1.0] * 48
    }
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print("  Base model loaded")

print("Step 2: Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
print("  Adapter loaded")

print("Step 3: Merging...")
model = model.merge_and_unload()
print("  Merged successfully")

print("Step 4: Saving merged model...")
os.makedirs(OUTPUT, exist_ok=True)
model.save_pretrained(OUTPUT, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT)
print(f"  Saved to {OUTPUT}")
print("Done!")
