"""Assemble per-category line files into one inflation script bank file.

  .venv/bin/python scripts/build_script_bank_file.py DIR OUT.txt

DIR holds <category>.txt files (one line per script line). Every line is checked against the existing
bank and against the other files (a line may appear in only one category), then OUT.txt is written in
the bank's "## category" format, sorted by category. Nothing in faily/data is modified unless OUT.txt is
a path there. Exit status is non-zero if any line was rejected (rejected lines are not written).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_script_lines import check  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, out = Path(argv[0]), Path(argv[1])
    from faily.modules.inflation import CATEGORIES
    # A previous build of OUT is part of the bank load_script_bank() reads; move it aside while checking,
    # otherwise every line would count as a duplicate of itself.
    aside = out.with_name(out.name + ".bak")
    if out.exists():
        out.replace(aside)
    try:
        return _build(src, out, CATEGORIES)
    except BaseException:
        if aside.exists() and not out.exists():
            aside.replace(out)
        raise
    finally:
        aside.unlink(missing_ok=True) if out.exists() else None


def _build(src: Path, out: Path, CATEGORIES) -> int:
    sections, rejected = [], 0
    seen: set[str] = set()
    for cat in CATEGORIES:
        f = src / f"{cat}.txt"
        if not f.exists():
            print(f"  ! no file for {cat}")
            continue
        lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        # drop anything another category already claimed in this build
        fresh = [l for l in lines if " ".join(l.split()).casefold() not in seen]
        rejected += len(lines) - len(fresh)
        problems, stats = check(cat, fresh)
        rejected += len(problems)
        for l in stats["good"]:
            seen.add(" ".join(l.split()).casefold())
        print(f"  {cat:11s} {len(stats['good']):4d} lines  ({len(lines) - len(fresh)} cross-category dupes, {len(problems)} rejected)")
        sections.append((cat, stats["good"]))
    body = ["# Extra script lines — ~300 per category, written to reduce overlap when inflating one category.",
            "# Same format as inflation_scripts.txt; merged by load_script_bank(). Checked with scripts/check_script_lines.py.", ""]
    for cat, lines in sections:
        body += [f"## {cat}", *lines, ""]
    out.write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {out} ({sum(len(l) for _, l in sections)} lines)")
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
