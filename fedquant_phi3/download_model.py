from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Starting download...")

model_name = "microsoft/Phi-3-mini-4k-instruct"

# Download tokenizer
print("Step 1: Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer downloaded \n")

# Download model
print("Step 2: Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
print("Model downloaded \n")

# Save to your models folder
print("Step 3: Saving to models folder...")
tokenizer.save_pretrained("./models/phi3-mini")
model.save_pretrained("./models/phi3-mini")
print("Saved to ./models/phi3-mini \n")

print("Phi-3-mini is ready!")