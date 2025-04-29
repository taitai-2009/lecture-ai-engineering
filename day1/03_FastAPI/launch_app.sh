#!/usr/bin/bash
set -a            # 以降 source した変数を自動 export
source ../.env    # bash の子シェルに一気に読み込み
set +a

# ─── HuggingFace CLI へのログイン ───────────────────
# （一度だけで ~/.huggingface/token にキャッシュ）
huggingface-cli login --token "$HUGGINGFACE_TOKEN"

# .streamlit/secrets.toml を here-doc で作成 ─────────────────
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<EOF
[huggingface]
token = "${HUGGINGFACE_TOKEN}"
EOF

python3 app.py
