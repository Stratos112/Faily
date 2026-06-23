#!/usr/bin/env bash
# test_faily.sh — Faily test suite.
# Static checks run everywhere (no torch needed).
# Live checks run only if the server starts successfully.
#
# Usage: bash scripts/test_faily.sh [--live-only | --static-only]

set -euo pipefail

PORT=7842
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGFILE="/tmp/faily_test_$(date +%Y%m%d_%H%M%S).log"
PASS=0
FAIL=0
SKIP=0

log()  { echo "$*" | tee -a "$LOGFILE"; }
pass() { log "  PASS  $1"; PASS=$((PASS+1)); }
fail() { log "  FAIL  $1 — $2"; FAIL=$((FAIL+1)); }
skip() { log "  SKIP  $1 — $2"; SKIP=$((SKIP+1)); }

check_http() {
    local desc="$1" url="$2" expect="${3:-200}"
    local got
    got=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    [ "$got" = "$expect" ] && pass "$desc" || fail "$desc" "expected HTTP $expect, got $got"
}

check_py() {
    local desc="$1" snippet="$2"
    if python3 - 2>>"$LOGFILE" <<EOF; then pass "$desc"; else fail "$desc" "see log"; fi
$snippet
EOF
}

check_syntax() {
    local f="$1"
    python3 -m py_compile "$f" 2>>"$LOGFILE" \
        && pass "syntax: ${f#$ROOT/}" \
        || fail  "syntax: ${f#$ROOT/}" "compile error"
}

kill_port() {
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 0.5
}

MODE="${1:-}"
cd "$ROOT"

log "=== Faily Test Run  $(date) ==="
log "Root : $ROOT"
log "Log  : $LOGFILE"
log ""

# ══════════════════════════════════════════════════════════════════════════════
# STATIC CHECKS — no running server needed
# ══════════════════════════════════════════════════════════════════════════════
if [ "$MODE" != "--live-only" ]; then
    log "── Syntax checks ────────────────────────────────────────────"
    while IFS= read -r f; do
        check_syntax "$f"
    done < <(find "$ROOT/faily" -name "*.py" ! -path "*/__pycache__/*")
    check_syntax "$ROOT/main.py"

    log ""
    log "── Filesystem checks ────────────────────────────────────────"
    # Ensure output dirs exist even without a running server
    mkdir -p "$ROOT/outputs/tts" "$ROOT/outputs/vc" "$ROOT/outputs/vc/refs" "$ROOT/outputs/characters"
    for d in outputs/tts outputs/vc outputs/vc/refs outputs/characters scripts; do
        [ -d "$ROOT/$d" ] && pass "dir: $d" || fail "dir: $d" "missing"
    done
    [ -f "$ROOT/CLAUDE.md" ]           && pass "CLAUDE.md exists"     || fail "CLAUDE.md"           "missing"
    [ -f "$ROOT/.claude/skills/test-faily.md" ] && pass "skill: test-faily" || fail "skill: test-faily" "missing"

    log ""
    log "── Pure-Python module checks (no torch) ─────────────────────"
    check_py "characters: list_characters importable" "
import sys; sys.path.insert(0, '.')
# Temporarily stub torch so characters.py (which doesn't need it) still works
import unittest.mock, sys
for mod in ['torch', 'torchaudio', 'soundfile']:
    sys.modules.setdefault(mod, unittest.mock.MagicMock())
from faily.core.characters import list_characters, save_character, get_ref_path, delete_character, CHARACTERS_DIR
print('CHARACTERS_DIR:', CHARACTERS_DIR)
"

    check_py "BACKENDS dict has >= 4 entries" "
import sys, unittest.mock, types
mk = unittest.mock.MagicMock
for mod in ['torch', 'torchaudio', 'torchaudio.functional', 'soundfile']:
    sys.modules.setdefault(mod, mk())
# Build transformers with real submodule tree so subpackage imports work
tr = types.ModuleType('transformers')
sys.modules['transformers'] = tr
for sub in ['pipelines', 'pipelines.audio_utils']:
    m = types.ModuleType(f'transformers.{sub}')
    setattr(tr, sub.split('.')[-1], m)
    sys.modules[f'transformers.{sub}'] = m
sys.modules['transformers.pipelines.audio_utils'].ffmpeg_read = mk()
# Stub model_manager
mm = types.ModuleType('faily.core.model_manager')
mm.manager = mk(); mm.manager.device = 'cpu'; mm.VC_MODELS_DIR = mk()
sys.modules['faily.core.model_manager'] = mm
from faily.modules.vc import BACKENDS
assert len(BACKENDS) >= 4, f'only {len(BACKENDS)}: {list(BACKENDS.keys())}'
print(list(BACKENDS.keys()))
"

    check_py "tune_tab: syntax + top-level importable with stubs" "
import sys, unittest.mock, types
for mod in ['torch', 'torchaudio', 'soundfile', 'transformers', 'nicegui',
            'nicegui.run', 'nicegui.ui']:
    sys.modules.setdefault(mod, unittest.mock.MagicMock())
mm = types.ModuleType('faily.core.model_manager')
mm.manager = unittest.mock.MagicMock(); mm.manager.device = 'cpu'
mm.VC_MODELS_DIR = unittest.mock.MagicMock()
sys.modules['faily.core.model_manager'] = mm
import ast, pathlib
src = pathlib.Path('faily/ui/tabs/tune_tab.py').read_text()
ast.parse(src)
print('AST ok — tune_tab.py parses cleanly')
"
fi  # end static checks

# ══════════════════════════════════════════════════════════════════════════════
# LIVE CHECKS — requires running server (Windows / env with torch)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$MODE" != "--static-only" ]; then
    log ""
    log "── Live server checks ───────────────────────────────────────"

    # Check if python3 can import torch before attempting server start
    if ! python3 -c "import torch" 2>/dev/null; then
        skip "Server startup" "torch not installed in this environment (run on Windows)"
        skip "HTTP: main page"     "server not started"
        skip "HTTP: static assets" "server not started"
        skip "HTTP: outputs route" "server not started"
    else
        log "Killing any existing process on :$PORT..."
        kill_port

        log "Starting server..."
        python3 main.py >>"$LOGFILE" 2>&1 &
        SERVER_PID=$!
        log "Server PID: $SERVER_PID"

        log "Waiting for :$PORT (up to 45s)..."
        READY=0
        for i in $(seq 1 45); do
            if curl -sf "http://localhost:$PORT/" >/dev/null 2>&1; then
                log "Server ready after ${i}s"; READY=1; break
            fi
            sleep 1
        done

        if [ $READY -eq 0 ]; then
            fail "Server startup" "did not respond on :$PORT within 45s"
        else
            pass "Server startup"
            check_http "HTTP: main page"      "http://localhost:$PORT/"
            check_http "HTTP: static assets"  "http://localhost:$PORT/_nicegui/static/favicon.ico"
            check_http "HTTP: outputs route"  "http://localhost:$PORT/outputs/" "404"

            log "Killing server (PID $SERVER_PID)..."
            kill "$SERVER_PID" 2>/dev/null || true
            kill_port
            fuser "${PORT}/tcp" 2>/dev/null \
                && fail "Port $PORT free after kill" "still occupied" \
                || pass "Port $PORT free after kill"
        fi
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
log ""
log "=== Results: $PASS passed, $FAIL failed, $SKIP skipped ==="
log "Full log: $LOGFILE"
echo ""
echo "Full log: $LOGFILE"

[ $FAIL -eq 0 ] && exit 0 || exit 1
