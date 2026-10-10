#!/bin/bash
# ==============================================================================
# Portable XM MT5 Setup Script for Gold Autonomous Trading System
# Downloads pre-configured XM Global MT5 (Wine distribution) from GitHub Release
# and extracts it directly into the Wine prefix on any Linux host.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MT5_ZIP="$ROOT_DIR/mt5.zip"
RELEASE_URL="https://github.com/sljeevan/gold-trading-system/releases/download/v1.0.0/mt5.zip"

echo "============================================================"
echo "🚀 Portable XM MT5 Setup for Gold Trading System"
echo "============================================================"

if [ -f "$MT5_ZIP" ]; then
    echo "✅ Found existing local archive: $MT5_ZIP ($(ls -lh "$MT5_ZIP" | awk '{print $5}'))"
else
    echo "⬇️ Downloading mt5.zip from GitHub Release..."
    echo "   URL: $RELEASE_URL"
    curl -fL --progress-bar -o "$MT5_ZIP" "$RELEASE_URL"
    echo "✅ Download completed: $MT5_ZIP ($(ls -lh "$MT5_ZIP" | awk '{print $5}'))"
fi

WINE_PROGRAM_FILES="$HOME/.wine/drive_c/Program Files"
if [ -d "$HOME/.wine" ]; then
    echo "📂 Extracting XM Global MT5 to Wine directory ($WINE_PROGRAM_FILES)..."
    mkdir -p "$WINE_PROGRAM_FILES"
    unzip -q -o "$MT5_ZIP" -d "$WINE_PROGRAM_FILES"
    echo "✅ XM Global MT5 installed successfully at $WINE_PROGRAM_FILES/XM Global MT5"
else
    echo "ℹ️ Wine directory (~/.wine) not detected."
    echo "   Archive is saved at $MT5_ZIP."
    echo "   Once Wine is initialized, run this script again to extract."
fi

echo "============================================================"
echo "🎯 Setup complete!"
echo "============================================================"
