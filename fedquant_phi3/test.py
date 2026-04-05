import torch
print("PyTorch installed:", torch.__version__)
print("GPU detected:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

import transformers
print("Transformers installed:", transformers.__version__)

import peft
print("PEFT installed:", peft.__version__)

import flwr
print("Flower installed:", flwr.__version__)

print("\nEverything is working!")