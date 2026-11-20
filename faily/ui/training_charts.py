"""Progress graphs for a Piper training run (piper.training_history): the loss, its parts,
pronunciation and the voice_metrics measures, one at a time behind a selector. Used live in
the training dialog and, from the copy saved with a model, in the MODELS tab's DETAILS."""
from nicegui import ui

from faily.modules.piper import LOSS_PARTS
from faily.modules.voice_metrics import METRICS

_AMBER, _BLUE, _GREY, _GREEN = "#f59e0b", "#38bdf8", "#a3a3a3", "#34d399"
_PART_COLOURS = ["#f59e0b", "#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#94a3b8"]

# key -> (button label, what the graph shows)
CHARTS: dict[str, tuple[str, str]] = {
    "loss": ("LOSS", "Generator (amber) and validation (blue) loss; faint = per epoch, bold = 10-epoch average. "
             "Lower is better, but it flattens long before the voice stops changing — check the measures."),
    # one graph per loss term, each on its own scale (they differ by 10-50x)
    **{tag: (label.upper(), f"Loss part: {what}. Faint = per epoch, bold = 5-epoch average. "
             "Only runs trained after the parts were logged have these.")
       for tag, (label, what) in LOSS_PARTS.items()},
    "pronunciation": ("PRONUNCIATION", "Share of words Whisper hears right when the model reads 16 sentences "
                      "it never trained on (%). Scored as each checkpoint comes in."),
    **{k: (v[0], v[2] + (f" ({v[1]})" if v[1] else "")) for k, v in METRICS.items()},
}
_UNITS = {"pronunciation": "%", **{k: v[1] for k, v in METRICS.items()}}


def smooth(xs: list, window: int = 10) -> list:
    """Trailing mean, skipping gaps (None)."""
    out = []
    for i in range(len(xs)):
        vals = [v for v in xs[max(0, i - window + 1):i + 1] if v is not None]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def _base(title: str = "") -> dict:
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": "#888", "fontFamily": "monospace"},
        "animation": False,
        "title": {"text": title, "left": "center", "top": "middle",
                  "textStyle": {"color": "#555", "fontSize": 11, "fontWeight": "normal"}},
        "grid": {"left": 52, "right": 70, "top": 28, "bottom": 28},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "value", "scale": True, "name": "epoch", "nameLocation": "middle",
                  "nameGap": 22, "splitLine": {"show": False}, "minInterval": 1},
    }


def _saved_line(mark_epoch: int | None) -> list:
    return [{"xAxis": mark_epoch, "lineStyle": {"color": "#e5e5e5", "type": "dashed", "width": 1},
             "label": {"formatter": "saved", "color": "#e5e5e5"}}] if mark_epoch is not None else []


def loss_chart_options(loss: dict, mark_epoch: int | None = None) -> dict:
    """ECharts options for a run's total loss (piper.read_loss_history). mark_epoch draws a
    labelled vertical line there (the epoch a saved model was taken from)."""
    ep = loss.get("epochs") or []

    def series(name, data, color, bold):
        return {
            "name": name, "type": "line", "showSymbol": False, "connectNulls": True,
            "data": [[e, None if v is None else round(v, 3)] for e, v in zip(ep, data)],
            "lineStyle": {"color": color, "width": 2 if bold else 1, "opacity": 1 if bold else 0.25},
            "itemStyle": {"color": color}, "silent": not bold,
        }

    mark = {"markLine": {"symbol": "none", "silent": True, "data": _saved_line(mark_epoch)}} if mark_epoch is not None else {}
    return {
        **_base("" if ep else "no loss logged yet"),
        "legend": {"show": True, "data": ["generator", "validation"], "textStyle": {"color": "#888"}, "top": 0},
        "yAxis": {"type": "value", "scale": True, "name": "", "splitLine": {"lineStyle": {"color": "#1a1a1a"}}},
        "series": [
            series("generator", loss.get("gen") or [], _AMBER, False),
            series("validation", loss.get("val") or [], _BLUE, False),
            # same names so the legend toggles raw + smoothed together
            {**series("generator", smooth(loss.get("gen") or []), _AMBER, True), **mark},
            series("validation", smooth(loss.get("val") or []), _BLUE, True),
        ],
    }


PART_WINDOW = 5      # epochs averaged into a loss part's bold line


def _part_options(loss: dict, tag: str, mark_epoch: int | None) -> dict:
    """One loss term on its own linear scale: per epoch (faint) + PART_WINDOW-epoch average (bold)."""
    ep, vals = loss.get("epochs") or [], (loss.get("parts") or {}).get(tag) or []
    colour = _PART_COLOURS[list(LOSS_PARTS).index(tag) % len(_PART_COLOURS)]

    def series(data, bold):
        return {
            "name": LOSS_PARTS[tag][0], "type": "line", "showSymbol": False, "connectNulls": True,
            "data": [[e, None if v is None else round(v, 4)] for e, v in zip(ep, data)],
            "lineStyle": {"color": colour, "width": 2 if bold else 1, "opacity": 1 if bold else 0.25},
            "itemStyle": {"color": colour}, "silent": not bold,
        }

    bold = series(smooth(vals, PART_WINDOW), True)
    if mark_epoch is not None:
        bold["markLine"] = {"symbol": "none", "silent": True, "data": _saved_line(mark_epoch)}
    return {
        **_base("" if vals else "not logged for this run"),
        "legend": {"show": False},
        "yAxis": {"type": "value", "scale": True, "name": "", "splitLine": {"lineStyle": {"color": "#1a1a1a"}}},
        "series": [series(vals, False), bold],
    }


def _metric_options(history: dict, key: str, mark_epoch: int | None) -> dict:
    rows = history.get("checkpoints") or []
    pts = [[r["epoch"], r[key]] for r in rows if r.get(key) is not None]
    ref = history.get("reference") or {}
    base, target = ref.get("base") or {}, ref.get("target") or {}
    unit = _UNITS.get(key, "")
    lines, values = [], [v for _, v in pts]
    if base.get(key) is not None:
        lines.append({"yAxis": base[key], "lineStyle": {"color": _GREY, "type": "dashed", "width": 1},
                      "label": {"formatter": f"{base.get('label') or 'base'}", "color": _GREY}})
        values.append(base[key])
    if target.get(key) is not None:
        lines.append({"yAxis": target[key], "lineStyle": {"color": _GREEN, "type": "dashed", "width": 1},
                      "label": {"formatter": "your clips", "color": _GREEN}})
        values.append(target[key])
    lines += _saved_line(mark_epoch)
    axis = {"type": "value", "name": unit, "splitLine": {"lineStyle": {"color": "#1a1a1a"}}}
    if values:                          # keep the reference lines on screen, not just the points
        lo, hi = min(values), max(values)
        pad = (hi - lo) * 0.15 or max(abs(hi) * 0.05, 1.0)
        axis.update(min=round(lo - pad, 2), max=round(hi + pad, 2))
    return {
        **_base("" if pts else "no checkpoints yet — measured as each one comes in"),
        "legend": {"show": False},
        "yAxis": axis,
        "series": [{
            "name": CHARTS[key][0].lower(), "type": "line", "data": pts, "showSymbol": True, "symbolSize": 6,
            "lineStyle": {"color": _AMBER, "width": 2}, "itemStyle": {"color": _AMBER},
            "markLine": {"symbol": "none", "silent": True, "data": lines},
        }],
    }


def chart_options(history: dict, key: str, mark_epoch: int | None = None) -> dict:
    """ECharts options for graph `key` (a CHARTS key) of a training history."""
    loss = history.get("loss") or {}
    if key == "loss":
        return loss_chart_options(loss, mark_epoch)
    if key in LOSS_PARTS:
        return _part_options(loss, key, mark_epoch)
    return _metric_options(history, key, mark_epoch)


# Generator modules a loss part can't improve through when they are frozen (MATCH on Base Model).
_PART_FROZEN = {"loss_dur": ("dp", "rhythm is on Base Model, so the duration predictor is frozen and this "
                             "can't go down — a flat line is expected")}


def describe(history: dict, key: str) -> str:
    """The text under graph `key`: what it shows, plus a note when this run's settings explain its shape."""
    text = CHARTS[key][1]
    module, note = _PART_FROZEN.get(key, (None, ""))
    frozen = ((history.get("run") or {}).get("frozen") or "").split(",")
    return f"{text} Here {note}." if module and module in frozen else text


# The selector: graphs in groups, each entry with its latest value (and the target in grey).
GROUPS: list[tuple[str, list[str]]] = [
    ("LOSS", ["loss", *LOSS_PARTS]),
    ("QUALITY", ["pronunciation", "match"]),
    ("VOICE", ["wps", "pitch", "pitch_range", "centroid", "loudness", "noise"]),
]
_NAMES = {"loss": "Total", **{t: v[0][0].upper() + v[0][1:] for t, v in LOSS_PARTS.items()},
          "pronunciation": "Pronunciation", "match": "Voice match",
          "wps": "Speaking rate", "pitch": "Pitch", "pitch_range": "Pitch range", "centroid": "Brightness",
          "loudness": "Loudness", "noise": "Background"}
_SHORT_UNITS = {"pronunciation": "%", "match": "%", "wps": "w/s", "pitch": "Hz", "pitch_range": "st",
                "centroid": "Hz", "loudness": "dB", "noise": "dB"}


def _fmt(v: float, key: str) -> str:
    if key in ("wps", "pitch_range"):
        return f"{v:.1f}"
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"


def latest(history: dict, key: str) -> tuple[str, str]:
    """(latest value, target) as short text for the selector; "" where there is none."""
    if key == "loss":
        gen = [v for v in smooth((history.get("loss") or {}).get("gen") or []) if v is not None]
        return (f"{gen[-1]:.1f}" if gen else "", "")
    if key in LOSS_PARTS:
        part = [v for v in smooth(((history.get("loss") or {}).get("parts") or {}).get(key) or [], PART_WINDOW)
                if v is not None]
        return (_fmt(part[-1], key) if part else "", "")
    vals = [r[key] for r in history.get("checkpoints") or [] if r.get(key) is not None]
    target = ((history.get("reference") or {}).get("target") or {}).get(key)
    unit = _SHORT_UNITS.get(key, "")
    now = f"{_fmt(vals[-1], key)} {unit}".strip() if vals else ""
    return now, (f"→ {_fmt(target, key)}" if target is not None and key != "match" else "")


def history_chart(history: dict | None = None, mark_epoch: int | None = None, height: int = 260):
    """Graph list (left, grouped, with latest values) + the chosen graph (right). Returns
    update(history) to redraw with new data; update.select(key) switches graph."""
    state = {"h": history or {}, "key": "loss", "chart": None}
    with ui.row().classes("w-full gap-3 flex-nowrap items-stretch"):
        nav = ui.column().classes("w-48 shrink-0 gap-0")
        with ui.column().classes("flex-grow min-w-0 gap-1"):
            box = ui.element("div").classes("w-full")
            desc = ui.label("").classes("text-[#555] font-mono text-[10px] leading-snug")

    def draw_nav():
        nav.clear()
        with nav:
            for group, keys in GROUPS:
                ui.label(group).classes("text-[#444] font-mono text-[9px] tracking-widest mt-1 mb-0.5 px-2")
                for key in keys:
                    on = key == state["key"]
                    now, target = latest(state["h"], key)
                    with ui.row().classes(
                        "w-full items-baseline gap-1 flex-nowrap px-2 py-0.5 cursor-pointer rounded-sm border-l-2 "
                        + ("border-amber-500 bg-[#1a1500]" if on else "border-transparent hover:bg-[#111]")
                    ) as row:
                        ui.label(_NAMES[key]).classes(
                            "font-mono text-[10px] flex-grow truncate " + ("text-amber-400" if on else "text-[#888]"))
                        if now:
                            ui.label(now).classes("font-mono text-[10px] shrink-0 " + ("text-amber-400" if on else "text-[#bbb]"))
                        if target:
                            ui.label(target).classes("font-mono text-[9px] text-[#4a4a4a] shrink-0")
                    row.on("click", lambda k=key: select(k))

    def build():
        box.clear()
        with box:
            state["chart"] = ui.echart(chart_options(state["h"], state["key"], mark_epoch)).classes(
                "w-full").style(f"height:{height}px")
        desc.set_text(describe(state["h"], state["key"]))

    def update(h: dict):
        state["h"] = h
        draw_nav()
        desc.set_text(describe(h, state["key"]))
        opts = chart_options(h, state["key"], mark_epoch)
        chart = state["chart"]
        if chart is None or len(opts["series"]) != len(chart.options.get("series", [])):
            build()
            return
        chart.options.clear()
        chart.options.update(opts)
        chart.update()

    def select(key: str):
        state["key"] = key
        draw_nav()
        build()

    update.select = select
    draw_nav()
    build()
    return update
