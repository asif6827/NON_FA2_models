#!/usr/bin/env bash
# Print the Hugging Face model directory for the SPRING step-84 actor.
set -euo pipefail
actor_dir="$1/actor"
model_dir="$2"
has_weights() {
    [[ -s "$1/config.json" ]] && [[ -s "$1/model.safetensors" || -s "$1/model.safetensors.index.json" || -s "$1/pytorch_model.bin" || -s "$1/pytorch_model.bin.index.json" ]]
}
if has_weights "$actor_dir/huggingface"; then
    printf '%s\n' "$actor_dir/huggingface"
    exit 0
fi
mkdir -p "$(dirname "$model_dir")"
exec 9>"${model_dir}.lock"
flock 9
if ! has_weights "$model_dir"; then
    [[ ! -e "$model_dir" ]] || { echo "Incomplete existing model: $model_dir" >&2; exit 1; }
    staging=$(mktemp -d "${model_dir}.tmp.XXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    python -m verl.model_merger merge --backend fsdp --local_dir "$actor_dir" --target_dir "$staging" >&2
    has_weights "$staging" || { echo "Missing merged weights in $staging" >&2; exit 1; }
    mv "$staging" "$model_dir"
fi
printf '%s\n' "$model_dir"
