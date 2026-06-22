from huggingface_hub import snapshot_download
from pathlib import Path
import os

repo_id = "meta-llama/Llama-3.2-3B-Instruct"
target = Path("/export/home/asifali/HF_cache/Llama-3.2-3B-Instruct")

path = snapshot_download(
    repo_id=repo_id,
    local_dir=target,
    local_dir_use_symlinks=False,
    token=os.getenv("HF_TOKEN"),
    resume_download=True,
    local_files_only=False,)

print(path)

print("Downloaded to:", path)

