from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
import torch

print("PHASE 3 — Testing Aggregated Adapter")

# ── Step 1: Load base model ────────────────────────
print("\n[Step 1] Loading base Phi-3-mini INT4...")

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
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)
print("Base model loaded")

# ── Step 2: Load aggregated adapter ───────────────
print("\n[Step 2] Loading aggregated adapter...")
model = PeftModel.from_pretrained(
    model,
    "./results/aggregated",
    is_trainable=False
)
print("Aggregated adapter loaded")

# ── Step 3: Test on legal questions ───────────────
print("\n[Step 3] Testing on legal questions...")

questions = [
    "What is Article 32 of the Indian Constitution about?",
    "What is the difference between cognizable and non-cognizable offence?",
    "Define the term 'consideration' in contract law.",
]

model.eval()

for i, question in enumerate(questions):
    print(f"\n  Q{i+1}: {question}")
    print(f"  Answer: ", end="")

    inputs = tokenizer(
        f"Instruction: {question}\nOutput:",
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    print(answer.strip())

print("TEST COMPLETE")
