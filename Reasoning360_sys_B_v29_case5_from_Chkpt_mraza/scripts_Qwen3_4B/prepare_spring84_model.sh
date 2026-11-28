#!/usr/bin/env bash
# Resolve the prepared Hugging Face checkpoint for the SPRING step-84 actor.
set -euo pipefail
for model_dir in "$1/actor/huggingface" "$2"; do
    [[ -r "$model_dir/config.json" && -s "$model_dir/config.json" ]] || continue
    for weights in model.safetensors model.safetensors.index.json pytorch_model.bin pytorch_model.bin.index.json; do
        if [[ -r "$model_dir/$weights" && -s "$model_dir/$weights" ]]; then
            printf '%s\n' "$model_dir"
            exit 0
        fi
    done
done
printf 'No prepared SPRING checkpoint found in %s or %s. Merge the step-84 actor weights before submitting.\n' "$1/actor/huggingface" "$2" >&2
exit 1
