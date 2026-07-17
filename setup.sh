#!/usr/bin/env bash
# One-time setup for subtranslate.py: clones + builds whisper.cpp, downloads the models
# it needs, and installs the optional TEN VAD engine. Safe to re-run - each step is
# skipped if its output already exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$SCRIPT_DIR/whisper.cpp"
WHISPER_COMMIT="080bbbe85230f624f0b52127f1ae1218247989f9"

echo "==> Checking required commands..."
for cmd in git cmake ffmpeg python3 pip3; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Missing required command: $cmd"; exit 1; }
done

if [ ! -d "$WHISPER_DIR" ]; then
    echo "==> Cloning whisper.cpp..."
    git clone https://github.com/ggml-org/whisper.cpp "$WHISPER_DIR"
    git -C "$WHISPER_DIR" checkout "$WHISPER_COMMIT"
else
    echo "==> whisper.cpp already present, skipping clone"
fi

if [ ! -x "$WHISPER_DIR/build/bin/whisper-cli" ]; then
    echo "==> Building whisper.cpp (Vulkan backend - works across GPU vendors via Mesa/RADV, not just NVIDIA)..."
    cmake -B "$WHISPER_DIR/build" -S "$WHISPER_DIR" -DGGML_VULKAN=1
    cmake --build "$WHISPER_DIR/build" --config Release -j
else
    echo "==> whisper-cli already built, skipping"
fi

if [ ! -f "$WHISPER_DIR/models/ggml-large-v3.bin" ]; then
    echo "==> Downloading large-v3 model (~2.9GB)..."
    bash "$WHISPER_DIR/models/download-ggml-model.sh" large-v3
else
    echo "==> large-v3 model already present, skipping"
fi

if [ ! -f "$WHISPER_DIR/models/ggml-silero-v6.2.0.bin" ]; then
    echo "==> Downloading silero VAD model..."
    bash "$WHISPER_DIR/models/download-vad-model.sh" silero-v6.2.0
else
    echo "==> silero VAD model already present, skipping"
fi

echo "==> Installing ten-vad (optional, only needed for --vad-engine ten)..."
pip3 install --user ten-vad || echo "  [WARNING] pip install ten-vad failed - --vad-engine ten won't work until this is resolved"

# ten-vad's native library needs libc++ (LLVM's C++ runtime); most distros ship GCC's
# libstdc++ by default and don't have this installed.
if ! ldconfig -p 2>/dev/null | grep -q "libc++\.so"; then
    echo ""
    echo "  [NOTE] libc++ runtime not found - required for --vad-engine ten to actually load."
    if command -v apt-get >/dev/null 2>&1; then
        echo "  Install it with: sudo apt-get install libc++1"
    elif command -v dnf >/dev/null 2>&1; then
        echo "  Install it with: sudo dnf install libcxx"
    elif command -v pacman >/dev/null 2>&1; then
        echo "  Install it with: sudo pacman -S libc++"
    else
        echo "  Install your distro's LLVM libc++ runtime package."
    fi
fi

echo ""
echo "==> Setup complete."
echo ""
echo "One manual step remains: install LM Studio (https://lmstudio.ai), load a model, and"
echo "start its local server (default: http://localhost:1234/v1) - subtranslate.py talks to"
echo "it as an OpenAI-compatible API and does not bundle or install it itself."
echo ""
echo "Try it: python3 subtranslate.py your_video.mp4 --lang ja"
