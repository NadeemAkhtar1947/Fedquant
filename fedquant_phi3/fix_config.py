import json

config_path = "./models/phi3-mini/config.json"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Only 3 fields allowed — type, short_factor, long_factor
config["rope_scaling"] = {
    "type": "longrope",
    "short_factor": [1.0] * 48,
    "long_factor": [1.0] * 48
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("config.json fixed")