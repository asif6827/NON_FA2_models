from huggingface_hub import snapshot_download
from pathlib import Path


repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
target = Path("/ibex/scratch/zakroum/lab-asif/HF_cache/Qwen2.5-0.5B-Instruct")  # your folder

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=target,
    local_dir_use_symlinks=False,   # real files (not symlinks), avoids mount issues
    resume_download=True,           # good for flaky networks
    local_files_only=False          # ensure it goes online
)
print("Downloaded to:", path)

#############################################################################