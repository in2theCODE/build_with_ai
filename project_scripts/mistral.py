# Download to a specific path with MPS support
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

# Get your API key - you need to create this in your .env file
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Set your desired download path
download_path = "/Users/justinrussell/.models"

# Make sure the directory exists
os.makedirs(download_path, exist_ok=True)

# Use a non-gated model instead
# The Mistral open models are available through other repositories
tokenizer = AutoTokenizer.from_pretrained(
    "TheBloke/Mistral-7B-v0.1-GGUF",  # Non-gated alternative
    cache_dir=download_path,
    token=api_key,  # Pass the token for authentication
    trust_remote_code=True
)

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Download DeepSeek model to the specific path and move to MPS
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base",
    cache_dir=download_path,
    token=api_key,  # Pass the token here too
    torch_dtype=torch.float16,  # Use half precision for better performance
    trust_remote_code=True
)

# Move model to MPS device
model = model.to(device)

print(f"Models successfully downloaded to {download_path}")