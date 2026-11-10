"""Check a batch of NEW inflation script lines before they join the bank.

  .venv/bin/python scripts/check_script_lines.py CATEGORY lines.txt [--write]

`lines.txt` holds one script line per line (no headers). Reported, per line:
  duplicate      same text (case/space-insensitive) as a line already in the bank or earlier in the batch
  length         outside the category's word range
  characters     anything the TTS/aligner pipeline shouldn't see (only letters, digits and  . , ; : ! ? ' " - — … $ % & ( ) /  are allowed)
  ending         doesn't end with . ! ? … (optionally followed by a quote or bracket)
  digits         a number format the force-aligner can't read aloud (spoken_tokens leaves digits behind)
and for the batch: how the lines START (the bank balances these, see inflation.START_WEIGHTS) and which
three-word openings repeat too often. Exit status is non-zero when any line has a problem.
--write drops the problem lines and rewrites the file with only the good ones.
"""
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# words per line allowed in each category
WORDS = {"pacing": (1, 45), "intonation": (1, 14), "numbers": (4, 18)}
DEFAULT_WORDS = (4, 16)
MAX_SAME_OPENING = 8                       # at most this many lines may share their first three words
_ALLOWED = re.compile(r"^[A-Za-z0-9 .,;:!?'\"\-—…$%&()/’“”]+$")
_ENDING = re.compile(r"[.!?…][\"')”’]*$")


def norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def check(category: str, lines: list[str]) -> tuple[list[tuple[str, str]], dict]:
    """([(line, problem)], stats). Imports the project lazily (it pulls in torch)."""
    from faily.modules.autoqualify import spoken_tokens
    from faily.modules.inflation import CATEGORIES, load_script_bank, start_class

    if category not in CATEGORIES:
        raise SystemExit(f"unknown category {category!r}; choose from {', '.join(CATEGORIES)}")
    seen = {norm(t): cat for cat, texts in load_script_bank().items() for t in texts}
    lo, hi = WORDS.get(category, DEFAULT_WORDS)
    problems: list[tuple[str, str]] = []
    good: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        key = norm(text)
        why = None
        if key in seen:
            why = f"duplicate (already in {seen[key]})" if seen[key] != "(this batch)" else "duplicate (earlier in this batch)"
        elif not (lo <= len(text.split()) <= hi):
            why = f"length: {len(text.split())} words (allowed {lo}-{hi})"
        elif not _ALLOWED.match(text):
            bad = sorted({c for c in text if not _ALLOWED.match(c)})
            why = f"characters: {bad}"
        elif not _ENDING.search(text):
            why = "ending: should end with . ! ? or …"
        else:
            left = [w for w, _ in spoken_tokens(text) if any(c.isdigit() for c in w)]
            if left:
                why = f"digits the aligner can't read: {left}"
        if why:
            problems.append((text, why))
        else:
            seen[key] = "(this batch)"
            good.append(text)
    starts = collections.Counter(start_class(t) for t in good)
    openings = collections.Counter(" ".join(re.findall(r"[A-Za-z']+", t.lower())[:3]) for t in good)
    stats = {
        "good": good, "starts": starts,
        "repeated_openings": [(o, n) for o, n in openings.most_common() if n > MAX_SAME_OPENING],
        "words": (min((len(t.split()) for t in good), default=0), max((len(t.split()) for t in good), default=0)),
    }
    return problems, stats


def main(argv: list[str]) -> int:
    write = "--write" in argv
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    category, path = args[0], Path(args[1])
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    problems, stats = check(category, lines)
    n = len(lines)
    print(f"{category}: {len(stats['good'])} good of {n} lines  (words {stats['words'][0]}-{stats['words'][1]})")
    total = max(len(stats["good"]), 1)
    print("  starts: " + ", ".join(f"{k} {v} ({v / total:.0%})" for k, v in sorted(stats["starts"].items(), key=lambda kv: -kv[1])))
    print("  target: vowel 23%, sonorant 28%, fricative 19%, the 10%, th 5%, stop 15%")
    for opening, count in stats["repeated_openings"]:
        print(f"  opening repeated {count}x: {opening!r}  (max {MAX_SAME_OPENING})")
    for text, why in problems[:60]:
        print(f"  ✗ {why}: {text}")
    if len(problems) > 60:
        print(f"  … and {len(problems) - 60} more")
    if write:
        path.write_text("\n".join(stats["good"]) + "\n", encoding="utf-8")
        print(f"  rewrote {path} with the {len(stats['good'])} good lines")
    return 1 if problems or stats["repeated_openings"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
