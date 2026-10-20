"""A clip's waveform with a draggable kept region, for the review dialogs.

Same look as the Edit tab's trim view (filled RMS waveform, amber kept region), but the
region is set with one two-handled slider directly under the waveform, and optional
red marks show where noises were found.
"""
from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from nicegui import ui

_H = 72            # waveform height, px
_COLS = 500        # RMS columns


@lru_cache(maxsize=64)
def _peaks(path: str, mtime: float) -> tuple[tuple[float, ...], float]:
    """(normalised RMS per column, duration in seconds). mtime keys the cache so an
    edited clip redraws."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = min(_COLS, max(1, len(data)))
    peaks = np.array([float(np.sqrt(np.mean(c ** 2))) if len(c) else 0.0 for c in np.array_split(data, n)])
    if peaks.max() > 0:
        peaks = peaks / peaks.max()
    return tuple(peaks.tolist()), len(data) / sr


def waveform_svg(path: Path) -> tuple[str, float]:
    """(svg markup, duration)."""
    peaks, dur = _peaks(str(path), Path(path).stat().st_mtime)
    n = len(peaks)
    W, mid, margin = 500, _H / 2, 4
    top = [(i * W / n + W / n / 2, mid - max(1.0, p * (mid - margin))) for i, p in enumerate(peaks)]
    bot = [(i * W / n + W / n / 2, mid + max(1.0, p * (mid - margin))) for i, p in enumerate(peaks)]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in top + list(reversed(bot)))
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {_H}" preserveAspectRatio="none" '
        f'width="100%" height="{_H}" style="position:absolute;top:0;left:0;">'
        f'<line x1="0" y1="{mid:.1f}" x2="{W}" y2="{mid:.1f}" stroke="rgba(80,80,80,0.25)" stroke-width="0.5"/>'
        f'<polygon points="{pts}" fill="rgba(160,120,40,0.55)"/></svg>'
    )
    return svg, dur


def region_picker(
    path: Path, start: float, end: float, *, marks: list[tuple[float, float]] | None = None,
    on_change=None,
):
    """Waveform + range slider. `marks` are (start, end) seconds drawn in red (noises).
    Returns an object with .start, .end (seconds) and .set(start, end).

    The shaded overlay is updated in the browser as the handles move; on_change(start, end)
    fires on the server side too."""
    svg, dur = waveform_svg(path)
    uid = "wf" + uuid.uuid4().hex[:8]
    dur = max(dur, 0.01)
    start, end = max(0.0, min(start, dur)), max(0.0, min(end, dur))
    pct = lambda t: max(0.0, min(100.0, t / dur * 100))
    mark_html = "".join(
        f'<div title="noise {a:.2f}–{b:.2f} s" style="position:absolute;top:0;height:100%;'
        f'left:{pct(a):.2f}%;width:{max(pct(b) - pct(a), 0.4):.2f}%;background:rgba(239,68,68,0.38);'
        f'border-left:1px solid rgba(239,68,68,0.7);pointer-events:none;"></div>'
        for a, b in (marks or [])
    )
    dim = "position:absolute;top:0;height:100%;background:rgba(0,0,0,0.62);pointer-events:none;"
    html = (
        f'<div style="position:relative;height:{_H}px;background:#0d0d0d;border-radius:3px;overflow:hidden;'
        f'border:1px solid #1a1a1a;">{svg}{mark_html}'
        f'<div id="{uid}-l" style="{dim}left:0;width:{pct(start):.2f}%;"></div>'
        f'<div id="{uid}-r" style="{dim}right:0;width:{100 - pct(end):.2f}%;"></div>'
        f'<div id="{uid}-k" style="position:absolute;top:0;height:100%;left:{pct(start):.2f}%;'
        f'width:{pct(end) - pct(start):.2f}%;border-left:2px solid rgba(245,158,11,0.8);'
        f'border-right:2px solid rgba(245,158,11,0.8);pointer-events:none;"></div></div>'
    )
    ui.html(html, sanitize=False).classes("w-full")
    label = ui.label("").classes("text-[#c9a227] font-mono text-[10px]")
    slider = ui.range(min=0, max=round(dur, 2), step=0.01, value={"min": round(start, 2), "max": round(end, 2)}).props(
        "color=amber dense"
    ).classes("w-full")

    def refresh(e=None):
        v = slider.value
        a, b = float(v["min"]), float(v["max"])
        label.set_text(f"keeping {a:.2f} – {b:.2f} s   ({b - a:.2f} s of {dur:.2f} s;  "
                       f"{a:.2f} s off the start, {dur - b:.2f} s off the end)")
        ui.run_javascript(
            f"(()=>{{const l=document.getElementById('{uid}-l'),r=document.getElementById('{uid}-r'),"
            f"k=document.getElementById('{uid}-k');if(!l)return;"
            f"l.style.width='{pct(a):.2f}%';r.style.width='{100 - pct(b):.2f}%';"
            f"k.style.left='{pct(a):.2f}%';k.style.width='{pct(b) - pct(a):.2f}%';}})()"
        )
        if on_change and e is not None:
            on_change(a, b)

    slider.on_value_change(refresh)
    refresh()

    class _Region:
        @property
        def start(self) -> float:
            return float(slider.value["min"])

        @property
        def end(self) -> float:
            return float(slider.value["max"])

        @property
        def duration(self) -> float:
            return dur

        def set(self, a: float, b: float) -> None:
            slider.set_value({"min": round(a, 2), "max": round(b, 2)})

    return _Region()
