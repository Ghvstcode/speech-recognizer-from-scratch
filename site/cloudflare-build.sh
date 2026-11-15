#!/bin/sh
# Cloudflare runs this on every push to main, then uploads _site/ with
# `npx wrangler deploy`. Its build machines have no Rust, so install it first.
set -eu
cd "$(dirname "$0")/.."

if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
        sh -s -- -y --profile minimal --target wasm32-unknown-unknown
    . "$HOME/.cargo/env"
fi
rustup target add wasm32-unknown-unknown

site/publish.sh --build
