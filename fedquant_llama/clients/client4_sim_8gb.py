import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW

print("CLIENT 4 — Simulated 8GB Phone (LLaMA 3.1 8B)")

print("\n[Step 1] Loading data...")
data = json.load(open("./data/clients/client4.json"))
print(f"Loaded {len(data)} examples")

print("\n[Step 2] Loading LLaMA 3.1 8B in BF16...")
tokenizer = AutoTokenizer.from_pretrained("./models/llama-3.1-8b", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "./models/llama-3.1-8b", torch_dtype=torch.bfloat16,
    trust_remote_code=True, attn_implementation="eager"
)
model = model.cuda()
print(f"Model loaded — GPU RAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

print("\n[Step 3] Attaching LoRA adapter r=16...")
lora_config = LoraConfig(r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable parameters: {trainable:,} out of {total:,}")

import os as _os
_prev = _os.environ.get("FEDQUANT_PREV_ADAPTER", "")
if _prev and _os.path.exists(_prev):
    import torch as _torch
    from safetensors.torch import load_file as _load_file
    from peft import set_peft_model_state_dict as _set_peft
    _weights = _load_file(f"{_prev}/adapter_model.safetensors")
    _target_rank = 16
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
    print(f"  Loaded + padded adapter to r=16")
else:
    print("  Round 1 - starting from base model")

print("\n[Step 4] Preparing training data...")
texts = [f"Instruction: {ex.get('instruction','')}\nInput: {ex.get('input','')}\nOutput: {ex.get('output','')}" for ex in data]
print(f"{len(texts)} examples formatted")

print("\n[Step 5] Training (1 FL round = 100 steps)...")
model.train()
optimizer = AdamW(model.parameters(), lr=2e-4)
for step, text in enumerate(texts[:100]):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    loss = model(**inputs, labels=inputs["input_ids"], use_cache=False).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"  Step {step+1}/100 — Loss: {loss.item():.4f}")

print("\n[Step 6] Saving LoRA adapter...")
model.save_pretrained("./results/adapters/client4")
print("CLIENT 4 DONE")
