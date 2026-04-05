import json
import os

# Path to by_task folder 
BY_TASK_PATH = "./data/by_task"

# Load each task file
print("Loading task files...\n")

def load_task(filename):
    path = os.path.join(BY_TASK_PATH, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {filename}: {len(data)} examples loaded")
    return data

qa            = load_task("qa_legal_reasoning.json")
section       = load_task("section_understanding.json")
definitions   = load_task("definitions.json")
summarization = load_task("summarization.json")
structural    = load_task("structural.json")
metadata      = load_task("metadata.json")
classification= load_task("classification.json")
ner           = load_task("ner.json")
comparison    = load_task("comparison.json")
constitution  = load_task("constitution_qa.json")

# Assign tasks to each client
print("\nAssigning tasks to clients...")

client_data = {
    "client1": qa[:5000],                                    # real phone — small QA chunk
    "client2": section + definitions,                        # sim 6GB
    "client3": summarization + structural + metadata,        # sim 6GB
    "client4": classification + ner,                         # sim 8GB
    "client5": comparison + constitution + qa[5000:10000],   # sim 12GB
}

# Save each client's data
os.makedirs("./data/clients", exist_ok=True)

print("\nSaving...\n")
for client_name, data in client_data.items():
    path = f"./data/clients/{client_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  {client_name}: {len(data)} examples")

print("\nDone! Check your data/clients/ folder")