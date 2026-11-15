#!/bin/sh
# Builds the site and publishes it to Cloudflare.
#
#     site/publish.sh            build and publish
#     site/publish.sh --build    build only, into _site/
#
# Publishing needs Node (for `npx wrangler`) and a Cloudflare login:
# either `npx wrangler login` once, or CLOUDFLARE_API_TOKEN and
# CLOUDFLARE_ACCOUNT_ID in the environment.
set -eu
cd "$(dirname "$0")/.."

# Where links to source files point, and the Cloudflare Web Analytics token
# (public: it is written into every page).
export SITE_REPO_URL="${SITE_REPO_URL:-https://github.com/Ghvstcode/speech-recognizer-from-scratch}"
export SITE_ANALYTICS_TOKEN="${SITE_ANALYTICS_TOKEN:-17855ba8f98e4cc1be32f37d83bdcde6}"

cargo run --release --manifest-path site/Cargo.toml

if [ "${1:-}" != "--build" ]; then
    npx --yes wrangler@4 deploy
fi
