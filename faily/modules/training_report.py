"""What a Piper training run did, written to disk when it stops (finished, stopped early or failed)
so it can be looked at later without the app: outputs/characters/<name>/training_report/.
Replaced by the next run's report.

  summary.txt     run settings, loss trend, one line per checkpoint (pronunciation, voice match, …)
  epochs.csv      one row per epoch: losses and loss parts, plus the measures at checkpoint epochs
  history.json    piper.training_history() as the progress graphs use it
  status.json     the run's status file (settings, final state, error message)
  log.txt         the run's full log
  overview.png    every graph on one sheet
  <graph>.png     each graph on its own (loss, loss_mel, …, pronunciation, match, pitch, …)
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

REPORT_DIR = "training_report"

_LOSS_COLOURS = {"gen": "#d97706", "val": "#0284c7"}


def report_dir(char_dir: Path) -> Path:
    return char_dir / REPORT_DIR


def _smooth(xs: list, window: int) -> list:
    out = []
    for i in range(len(xs)):
        vals = [v for v in xs[max(0, i - window + 1):i + 1] if v is not None]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def _graphs() -> list[tuple[str, str, str]]:
    """(key, title, unit) for every graph, in the dialog's order."""
    from faily.modules.piper import LOSS_PARTS
    from faily.modules.voice_metrics import METRICS
    return ([("loss", "Loss (generator / validation)", "")]
            + [(t, f"Loss part: {v[0]}", "") for t, v in LOSS_PARTS.items()]
            + [("pronunciation", "Pronunciation", "%")]
            + [(k, v[0].capitalize(), v[1]) for k, v in METRICS.items()])


def _draw(ax, history: dict, key: str, title: str, unit: str) -> None:
    loss = history.get("loss") or {}
    ep = loss.get("epochs") or []
    from matplotlib.ticker import MaxNLocator

    saved = history.get("saved_epoch")
    frozen = ((history.get("run") or {}).get("frozen") or "").split(",")
    if key == "loss_dur" and "dp" in frozen:
        title += "  (duration predictor frozen: flat is expected)"
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch", fontsize=8)
    if unit:
        ax.set_ylabel(unit, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)

    def faint_and_smooth(vals, colour, label, window):
        pts = [(e, v) for e, v in zip(ep, vals) if v is not None]
        sm = [(e, v) for e, v in zip(ep, _smooth(vals, window)) if v is not None]
        if pts:
            ax.plot(*zip(*pts), color=colour, alpha=0.25, linewidth=0.8)
            ax.plot(*zip(*sm), color=colour, linewidth=1.8, label=label)

    if key == "loss":
        faint_and_smooth(loss.get("gen") or [], _LOSS_COLOURS["gen"], "generator", 10)
        faint_and_smooth(loss.get("val") or [], _LOSS_COLOURS["val"], "validation", 10)
        if ep:
            ax.legend(fontsize=7)
    elif key.startswith("loss_"):
        faint_and_smooth((loss.get("parts") or {}).get(key) or [], "#7c3aed", None, 5)
    else:
        pts = [(r["epoch"], r[key]) for r in history.get("checkpoints") or [] if r.get(key) is not None]
        if pts:
            ax.plot(*zip(*pts), color="#d97706", marker="o", markersize=3, linewidth=1.6)
        ref = history.get("reference") or {}
        for side, colour, label in (("base", "#6b7280", None), ("target", "#059669", "your clips")):
            v = (ref.get(side) or {}).get(key)
            if v is not None:
                ax.axhline(v, color=colour, linestyle="--", linewidth=1,
                           label=label or (ref.get("base") or {}).get("label") or "base")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    if saved is not None:
        ax.axvline(saved, color="#111", linestyle=":", linewidth=1)
    if not ax.has_data():
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#999")


def _pngs(history: dict, out: Path) -> None:
    from matplotlib.figure import Figure      # OO API: no pyplot, no global backend state

    graphs = _graphs()
    for key, title, unit in graphs:
        fig = Figure(figsize=(8, 3.6), dpi=110, layout="constrained")
        _draw(fig.add_subplot(), history, key, title, unit)
        fig.savefig(out / f"{key}.png")
    cols = 4
    rows = -(-len(graphs) // cols)
    fig = Figure(figsize=(cols * 4.2, rows * 2.9), dpi=100, layout="constrained")
    for i, (key, title, unit) in enumerate(graphs, 1):
        _draw(fig.add_subplot(rows, cols, i), history, key, title, unit)
    fig.savefig(out / "overview.png")


def _csv(history: dict, out: Path) -> None:
    from faily.modules.piper import LOSS_PARTS
    from faily.modules.voice_metrics import METRICS

    loss = history.get("loss") or {}
    parts = loss.get("parts") or {}
    measures = ["pronunciation", *METRICS]
    by_epoch = {r["epoch"]: r for r in history.get("checkpoints") or []}
    epochs = sorted(set(loss.get("epochs") or []) | set(by_epoch))
    idx = {e: i for i, e in enumerate(loss.get("epochs") or [])}
    tags = [t for t in LOSS_PARTS if t in parts]

    def at(series, e):
        i = idx.get(e)
        return "" if i is None or i >= len(series or []) or series[i] is None else round(series[i], 4)

    with open(out / "epochs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "total_gen", "total_val", *tags, "checkpoint", *measures])
        for e in epochs:
            r = by_epoch.get(e)
            w.writerow([e, at(loss.get("gen"), e), at(loss.get("val"), e), *(at(parts[t], e) for t in tags),
                        "yes" if r else "", *(("" if not r or r.get(k) is None else r[k]) for k in measures)])


def _summary(history: dict, status: dict) -> str:
    from faily.modules.voice_metrics import METRICS

    run = status.get("run") or {}
    lines = [f"state: {status.get('state')}" + (f" — {status['message']}" if status.get("message") else "")]
    for k in ("base_voice_label", "base_epoch", "resumed_from", "max_epochs", "target_epoch", "learning_rate",
              "match", "frozen", "batch", "validation_split", "clips", "skipped_long"):
        if run.get(k) not in (None, ""):
            lines.append(f"{k}: {run[k]}")
    if status.get("epoch") is not None:
        lines.append(f"last epoch reached: {status['epoch']}")
    trend = (history.get("loss") or {}).get("trend")
    if trend:
        lines.append(f"loss trend: {trend}")
    keys = ["pronunciation", *METRICS]
    ref = history.get("reference") or {}
    lines += ["", "checkpoints (base / target rows are the reference lines):",
              "  " + "epoch".ljust(8) + "".join(k.rjust(15) for k in keys)]

    def row(name, d):
        return "  " + str(name).ljust(8) + "".join(
            ("" if d.get(k) is None else f"{d[k]:.2f}").rjust(15) for k in keys)

    if ref.get("base"):
        lines.append(row("base", ref["base"]))
    for r in history.get("checkpoints") or []:
        lines.append(row(r["epoch"], r))
    if ref.get("target"):
        lines.append(row("target", ref["target"]))
    return "\n".join(lines) + "\n"


def write_report(char_dir: Path) -> Path:
    """Write the report for the character's latest run (replacing any earlier one) and return
    its folder. Blocking (TensorBoard parsing, matplotlib); call from a thread."""
    from faily.modules import piper as P

    out = report_dir(char_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    history = P.training_history(char_dir)
    status = P.read_training_status(char_dir) or {}
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "status.json").write_text(json.dumps(status, indent=1))
    log = P.training_log_path(char_dir)
    if log.exists():
        shutil.copy2(log, out / "log.txt")
    (out / "summary.txt").write_text(_summary(history, status), encoding="utf-8")
    _csv(history, out)
    _pngs(history, out)
    return out
