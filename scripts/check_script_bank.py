"""Report duplicate lines in the buffer script bank (faily/data/buffer_scripts*.txt).

Duplicates are judged the same way sample_lines() does (whitespace- and
case-insensitive), within a category and across categories/files. Run after
adding lines:
    python scripts/check_script_bank.py          # report
    python scripts/check_script_bank.py --fix    # drop duplicate lines from the files
"""
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "faily" / "data"


def norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def main(fix: bool = False) -> int:
    seen: dict[str, str] = {}
    dupes = 0
    counts: dict[str, int] = {}
    files = [DATA / "buffer_scripts.txt", *sorted(DATA.glob("buffer_scripts_*.txt"))]
    for file in files:
        out, current = [], None
        for raw in file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("##"):
                current = line[2:].strip()
            elif line and not line.startswith("#") and current:
                key = norm(line)
                if key in seen:
                    dupes += 1
                    print(f"DUPLICATE in {file.name} [{current}]: {line}\n    first seen: {seen[key]}")
                    if fix:
                        continue
                else:
                    seen[key] = f"{file.name} [{current}]"
                    counts[current] = counts.get(current, 0) + 1
            out.append(raw)
        if fix:
            file.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("unique lines per category:", counts)
    print(f"{dupes} duplicate line(s)")
    return 1 if dupes and not fix else 0


if __name__ == "__main__":
    sys.exit(main("--fix" in sys.argv))
