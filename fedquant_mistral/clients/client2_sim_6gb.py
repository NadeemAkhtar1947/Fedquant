import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW

print("=" * 50)
print("CLIENT 2 — Simulated 6GB Phone (Mistral-7B)")
print("=" * 50)

# ── Step 1: Load Data ──────────────────────────────
print("\n[Step 1] Loading data...")
data = json.load(open("./data/clients/client2.json"))
print(f"Loaded {len(data)} examples")

# ── Step 2: Load Mistral-7B in BF16 ───────────────
print("\n[Step 2] Loading Mistral-7B in BF16...")
tokenizer = AutoTokenizer.from_pretrained(
    "./models/mistral-7b",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "./models/mistral-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="eager"
)
model = model.cuda()
print(f"Model loaded")
mem = torch.cuda.memory_allocated() / 1024**3
print(f"  GPU RAM used: {mem:.2f} GB")

# ── Step 3: Attach LoRA Adapter ───────────────────
print("\n[Step 3] Attaching LoRA adapter...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    _weights = load_file(f"{_prev}/adapter_model.safetensors")
    set_peft_model_state_dict(model, _weights)
    print(f"  Loaded adapter from round: {_os.environ.get('FEDQUANT_ROUND', '?')}")
else:
    print("  Round 1 - starting from base model")

# ── Step 4: Prepare Training Data ─────────────────
print("\n[Step 4] Preparing training data...")
def format_example(example):
    instruction = example.get("instruction", "")
    input_text  = example.get("input", "")
    output      = example.get("output", "")
    if input_text:
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
    return f"Instruction: {instruction}\nOutput: {output}"

texts = [format_example(ex) for ex in data]
print(f"{len(texts)} examples formatted")

# ── Step 5: Train ──────────────────────────────────
print("\n[Step 5] Training (1 FL round = 100 steps)...")
model.train()
optimizer = AdamW(model.parameters(), lr=2e-4)

for step, text in enumerate(texts[:100]):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=256, padding=True
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs, labels=inputs["input_ids"], use_cache=False)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"  Step {step+1}/140 — Loss: {loss.item():.4f}")

# ── Step 6: Save Adapter ───────────────────────────
print("\n[Step 6] Saving LoRA adapter...")
model.save_pretrained("./results/adapters/client2")
print("Saved to ./results/adapters/client2")
print("\n" + "=" * 50)
print("CLIENT 2 DONE — Adapter ready to send to server")
print("=" * 50)
