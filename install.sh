#!/usr/bin/env bash
# Aura install script — makes the `aura` command available system-wide.
#
# Usage:
#   ./install.sh            # Install using pip (preferred)
#   ./install.sh --symlink  # Symlink the shell wrapper to /usr/local/bin instead
#
# After install, just type `aura` from any directory to start.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RED='\033[91m'
GREEN='\033[92m'
CYAN='\033[96m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${CYAN}[aura]${RESET} $*"; }
ok()    { echo -e "${GREEN}[aura]${RESET} $*"; }
err()   { echo -e "${RED}[aura]${RESET} $*" >&2; }

cd "$SCRIPT_DIR"

if [[ "${1:-}" == "--symlink" ]]; then
    # Option B: symlink the shell wrapper into /usr/local/bin
    WRAPPER="$SCRIPT_DIR/aura"
    TARGET="/usr/local/bin/aura"

    if [[ ! -x "$WRAPPER" ]]; then
        err "Shell wrapper not found at $WRAPPER"
        exit 1
    fi

    info "Creating symlink: $TARGET -> $WRAPPER"
    if [[ -e "$TARGET" || -L "$TARGET" ]]; then
        info "Removing existing $TARGET"
        sudo rm -f "$TARGET"
    fi
    sudo ln -s "$WRAPPER" "$TARGET"
    ok "Symlink created."
else
    # Option A (preferred): pip editable install
    info "Installing Aura in editable mode via pip..."
    pip install -e . 2>&1 | while IFS= read -r line; do
        echo -e "  ${DIM}${line}${RESET}"
    done
    ok "pip install -e . complete."
fi

# Verify
echo ""
info "Verifying installation..."
if command -v aura &>/dev/null; then
    AURA_PATH="$(command -v aura)"
    ok "Found: $AURA_PATH"

    # Quick smoke test
    if aura --help &>/dev/null; then
        ok "aura --help works."
    else
        err "aura --help failed. Check the output above for errors."
        exit 1
    fi

    echo ""
    echo -e "${BOLD}Installation complete.${RESET}"
    echo -e "Type ${GREEN}aura${RESET} to start the Human Intelligence Engine."
else
    err "aura command not found on PATH after install."
    err "If using pip install, make sure your Python scripts directory is on PATH."
    err "Try:  export PATH=\"\$(python -m site --user-base)/bin:\$PATH\""
    exit 1
fi
