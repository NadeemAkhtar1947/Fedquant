from transformers import AutoModelForCausalLM
import torch

print("Loading Mistral-7B in BF16...")
model = AutoModelForCausalLM.from_pretrained(
    "./models/mistral-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("\nAttention layer names:")
for name, _ in model.named_modules():
    if any(x in name for x in ['q_proj','k_proj','v_proj','o_proj']):
        print(f"  {name}")
        break

print("\nAll unique layer types:")
seen = set()
for name, module in model.named_modules():
    parts = name.split('.')
    if len(parts) > 2:
        key = parts[-1]
        if key not in seen:
            seen.add(key)
            print(f"  {key}")
