# Download to a specific path with MPS support
import os

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


# Set your desired download path
download_path = "/Users/justinrussell/.models"

# Make sure the directory exists
os.makedirs(download_path, exist_ok=True)

# Download tokenizer to the specific path
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base", cache_dir=download_path, trust_remote_code=True
)

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Download model to the specific path and move to MPS
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base",
    cache_dir=download_path,
    torch_dtype=torch.float16,  # Use half precision for better performance
    trust_remote_code=True,
)

# Move model to MPS device
model = model.to(device)

# Now the model is downloaded to your specific path and ready to use with MPS
