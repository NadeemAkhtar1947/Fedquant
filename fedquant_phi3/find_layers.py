from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

config = AutoConfig.from_pretrained("./models/phi3-mini", trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "./models/phi3-mini",
    config=config,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

# Print all layer names
print("All layer names:")
for name, module in model.named_modules():
    if "proj" in name or "attn" in name:
        print(" ", name)