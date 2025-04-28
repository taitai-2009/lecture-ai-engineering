#!/usr/bin/bash
set -a            # 以降 source した変数を自動 export
source ../.env    # bash の子シェルに一気に読み込み
set +a

# ─── トラップを仕掛ける ────────────────────────────
# スクリプトが EXIT（正常終了 or Ctrl+C 等）したら cleanup 関数を呼ぶ
cleanup() {
  echo "🛑 停止中…"
  # jobs -pr で全バックグラウンドジョブの PID を取得して kill
  kill $(jobs -pr) 2>/dev/null || true
}
trap cleanup EXIT

# ngrok CLI にトークン登録（既にやってあれば上書きは harmless）
ngrok authtoken "$NGROK_TOKEN"

# ─── HuggingFace CLI へのログイン ───────────────────
# （一度だけで ~/.huggingface/token にキャッシュ）
huggingface-cli login --token "$HUGGINGFACE_TOKEN"

# .streamlit/secrets.toml を here-doc で作成 ─────────────────
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<EOF
[huggingface]
token = "${HUGGINGFACE_TOKEN}"
EOF

# Streamlit をバックグラウンド起動（HuggingFace トークンはアプリの中で os.getenv で参照可）
streamlit run app.py &
#  --server.address 0.0.0.0 \
#  --server.port    8501 &

# ngrok でトンネルを張る
ngrok http 8501
