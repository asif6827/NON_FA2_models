from huggingface_hub import snapshot_download
from pathlib import Path

# Llama 3.2 3B Instruct model
repo_id = "meta-llama/Llama-3.2-3B-Instruct"

# Change path as needed
target = Path("/export/home/asifali/HF_cache/Llama-3.2-3B-Instruct")

hf_token = "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"  # <-- your token here

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=target,
    local_dir_use_symlinks=False,   # store real files
    resume_download=True,           # resume if interrupted
    local_files_only=False          # force online download
)

print("Downloaded to:", path)
