#!/usr/bin/env bash
# Fetch the pinned mobile detector and Arabic recognizer. No pilot inputs are read.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target="$root/data/ingestion/phase3/setup/paddle_models"
fetch_model() {
    local name="$1" revision="$2" artifact
    local directory="$target/$name"
    mkdir -p "$directory"
    for artifact in inference.json inference.pdiparams inference.yml; do
        if [[ ! -s "$directory/$artifact" ]]; then
            curl --fail --location --silent --show-error --retry 2 --max-time 120 \
                "https://huggingface.co/PaddlePaddle/$name/resolve/$revision/$artifact?download=true" \
                --output "$directory/$artifact.partial"
            mv "$directory/$artifact.partial" "$directory/$artifact"
        fi
    done
}
fetch_model PP-OCRv5_mobile_det 0d63e78e2b680928f6b1747d76a08db6e645efb7
fetch_model arabic_PP-OCRv5_mobile_rec 33d91636a65dca87f5562cc48860332ae367ee1b
