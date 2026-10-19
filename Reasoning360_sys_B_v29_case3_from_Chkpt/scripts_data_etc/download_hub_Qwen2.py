from huggingface_hub import snapshot_download
from pathlib import Path

hp = False
panther = True
if hp:
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct"           # or "Qwen/Qwen3-1.7B-Instruct"
    target   = Path("/home/asif/data3/HF_Cache/Qwen2.5-0.5B-Instruct")  # your folder
elif panther:
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
    target = Path("/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct")  # your folder

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=target,
    local_dir_use_symlinks=False,   # real files (not symlinks), avoids mount issues
    resume_download=True,           # good for flaky networks
    local_files_only=False          # ensure it goes online
)
print("Downloaded to:", path)

