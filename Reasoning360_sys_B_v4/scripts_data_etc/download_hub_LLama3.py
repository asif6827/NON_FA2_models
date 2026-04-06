from huggingface_hub import snapshot_download
from pathlib import Path
import os

repo_id = "meta-llama/Llama-3.2-3B-Instruct"
target = Path("/export/home/asifali/HF_cache/Llama-3.2-3B-Instruct")
token = os.environ["HF_TOKEN"]

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=target,
    local_dir_use_symlinks=False,
    resume_download=True,
    token=token,
)

print("Downloaded to:", path)