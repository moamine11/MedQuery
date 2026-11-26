#!/usr/bin/env bash
# Fetch only the pinned CPU layout and fast table files. No pilot inputs are read.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target="$root/data/ingestion/phase3/setup/docling_models"
layout_repo=docling-project/docling-layout-heron
layout_rev=8f39ad3c0b4c58e9c2d2c84a38465abf757272d8
table_repo=docling-project/docling-models
table_rev=fc0f2d45e2218ea24bce5045f58a389aed16dc23
fetch() {
    local repository="$1" revision="$2" file="$3"
    local destination="$target/${repository//\//--}/$file"
    mkdir -p "$(dirname "$destination")"
    if [[ ! -s "$destination" ]]; then
        curl --fail --location --silent --show-error --retry 2 --max-time 240 \
            "https://huggingface.co/$repository/resolve/$revision/$file?download=true" \
            --output "$destination.partial"
        mv "$destination.partial" "$destination"
    fi
}
fetch "$layout_repo" "$layout_rev" config.json
fetch "$layout_repo" "$layout_rev" preprocessor_config.json
fetch "$layout_repo" "$layout_rev" model.safetensors &
layout_download=$!
fetch "$table_repo" "$table_rev" model_artifacts/tableformer/fast/tm_config.json
fetch "$table_repo" "$table_rev" model_artifacts/tableformer/fast/tableformer_fast.safetensors &
table_download=$!
wait "$layout_download"
wait "$table_download"
