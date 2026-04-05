from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading model from local folder...")

tokenizer = AutoTokenizer.from_pretrained("./models/phi3-mini")

model = AutoModelForCausalLM.from_pretrained(
    "./models/phi3-mini",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
)

print("Model loaded on GPU \n")

# Ask it a simple legal question
question = "What is a contract in simple terms?"

inputs = tokenizer(question, return_tensors="pt").to("cuda")

print("Asking:", question)
print("Thinking...\n")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Answer:", answer)
print("\nModel is working!")