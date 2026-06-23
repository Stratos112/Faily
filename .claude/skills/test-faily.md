# test-faily

Run Faily's test suite: start the server, verify HTTP + Python modules, kill everything cleanly.

## When to use
- After implementing new features to verify nothing is broken
- When the user asks to "test", "verify", or "run the app"
- Before committing significant changes

## How to run

```bash
cd /workspaces/projects/Faily && bash scripts/test_faily.sh
```

The script:
1. Kills any process on port 7842
2. Starts `python main.py` in background, logs to `/tmp/faily_test_<timestamp>.log`
3. Waits up to 45s for the server to be ready
4. Runs HTTP checks against `http://localhost:7842`
5. Runs Python module import checks
6. Verifies filesystem structure
7. Kills the server and confirms port 7842 is free
8. Prints PASS/FAIL summary

## Reading results

- Exit 0 = all checks passed
- Exit 1 = one or more failures — read the log for details
- Log path is printed at the end: `/tmp/faily_test_<timestamp>.log`

## Key checks

| Check | What it tests |
|---|---|
| HTTP 200 on `/` | Server starts and NiceGUI renders |
| `characters` module | `faily/core/characters.py` importable |
| `vc BACKENDS >= 4` | SpeechT5, XTTS v2, F5-TTS, Chatterbox all registered |
| `tune_tab` importable | TUNE tab wired correctly |
| `torchaudio` patched | torchcodec workaround active |
| `outputs/` dirs | All output subdirs created on startup |

## Notes

- The server uses port 7842. If another process holds it, the script kills it first.
- On Windows (user's machine), use the kill one-liner from README instead.
- Heavy model loading (XTTS, Chatterbox) is NOT triggered by these checks — they only test startup and imports.
