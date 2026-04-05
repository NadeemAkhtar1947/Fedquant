import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import os

print("CLIENT 5 — Simulated 12GB Phone")

# ── Step 1: Load Data ──────────────────────────────
print("\n[Step 1] Loading data...")
with open("./data/clients/client5.json", "r", encoding="utf-8") as f:
    data = json.load(f)
data = data
print(f"Loaded {len(data)} examples")

# ── Step 2: Load Compressed Model (BF16) ──────────
print("\n[Step 2] Loading Phi-3-mini in BF16 (compressed)...")
print("  Note: Client 5 uses 12GB RAM limit (high tier phone)")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

config = AutoConfig.from_pretrained(
    "./models/phi3-mini",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "./models/phi3-mini",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "./models/phi3-mini",
    config=config,
    torch_dtype=torch.bfloat16,
    
    trust_remote_code=True,
    attn_implementation="eager",
)

print(f"Model loaded")
mem = torch.cuda.memory_allocated() / 1024**3
print(f"  GPU RAM used: {mem:.2f} GB")
print(f"  Simulating: 12GB high tier phone")

# ── Step 3: Attach LoRA Adapter ───────────────────
print("\n[Step 3] Attaching LoRA adapter...")

lora_config = LoraConfig(
    r=32,                         # ← r=32 highest capacity for top tier phone
    lora_alpha=64,
    target_modules=["qkv_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable parameters: {trainable:,} out of {total:,}")


# Load previous round adapter if available
import os as _os
_prev = _os.environ.get("FEDQUANT_PREV_ADAPTER", "")
if _prev and _os.path.exists(_prev):
    import torch as _torch
    from safetensors.torch import load_file as _load_file
    from peft import set_peft_model_state_dict as _set_peft
    _weights = _load_file(f"{_prev}/adapter_model.safetensors")
    _target_rank = 32
    _padded = dict()
    for _k, _v in _weights.items():
        if "lora_A" in _k and _v.shape[0] < _target_rank:
            _pad = _torch.zeros(_target_rank - _v.shape[0], _v.shape[1])
            _padded[_k] = _torch.cat([_v, _pad], dim=0)
        elif "lora_B" in _k and _v.shape[1] < _target_rank:
            _pad = _torch.zeros(_v.shape[0], _target_rank - _v.shape[1])
            _padded[_k] = _torch.cat([_v, _pad], dim=1)
        else:
            _padded[_k] = _v
    _set_peft(model, _padded)
    print(f"  Loaded + padded adapter to r=32")
else:
    print("  Round 1 - starting from base model")

# ── Step 4: Prepare Training Data ─────────────────
print("\n[Step 4] Preparing training data...")

def format_example(example):
    instruction = example.get("instruction", "")
    input_text  = example.get("input", "")
    output      = example.get("output", "")
    if input_text:
        text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
    else:
        text = f"Instruction: {instruction}\nOutput: {output}"
    return text

texts = [format_example(ex) for ex in data]
print(f"{len(texts)} examples formatted")

# ── Step 5: Train for 1 Round ─────────────────────
print("\n[Step 5] Training (1 FL round = 3 steps)...")

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for step, text in enumerate(texts[:140]):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs, labels=inputs["input_ids"], use_cache=False)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step+1}/3 — Loss: {loss.item():.4f}")

# ── Step 6: Save LoRA Adapter ─────────────────────
print("\n[Step 6] Saving LoRA adapter...")
os.makedirs("./results/adapters", exist_ok=True)
model.save_pretrained("./results/adapters/client5")
print("Saved to ./results/adapters/client5")

print("CLIENT 5 DONE — Adapter ready to send to server")