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
    "parts": ("LOSS PARTS", "The generator loss split into its terms, plus the discriminator's (log scale). "
              "With rhythm or pronunciation on Base Model, the duration and prior terms can't go down. "
              "Only runs trained after this was added have the parts."),
    "pronunciation": ("PRONUNCIATION", "Share of words Whisper hears right when the model reads 16 sentences "
                      "it never trained on (%). Scored when the run ends."),
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


def _parts_options(loss: dict, mark_epoch: int | None) -> dict:
    ep, parts = loss.get("epochs") or [], loss.get("parts") or {}
    series = []
    for i, (tag, vals) in enumerate(parts.items()):
        label = LOSS_PARTS.get(tag, (tag, ""))[0]
        sm = smooth(vals, 5)
        series.append({
            "name": label, "type": "line", "showSymbol": False, "connectNulls": True,
            "data": [[e, round(v, 4) if v and v > 0 else None] for e, v in zip(ep, sm)],
            "lineStyle": {"color": _PART_COLOURS[i % len(_PART_COLOURS)], "width": 2},
            "itemStyle": {"color": _PART_COLOURS[i % len(_PART_COLOURS)]},
        })
    if series and mark_epoch is not None:
        series[0]["markLine"] = {"symbol": "none", "silent": True, "data": _saved_line(mark_epoch)}
    return {
        **_base("" if series else "no loss parts logged for this run"),
        "legend": {"show": True, "data": [s["name"] for s in series], "textStyle": {"color": "#888"}, "top": 0},
        "yAxis": {"type": "log", "name": "", "splitLine": {"lineStyle": {"color": "#1a1a1a"}}},
        "series": series or [{"type": "line", "data": []}],
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
    empty = ("scored when the run ends" if key == "pronunciation" else "measured as each checkpoint comes in")
    return {
        **_base("" if pts else f"no checkpoints yet — {empty}"),
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
    if key == "parts":
        return _parts_options(loss, mark_epoch)
    return _metric_options(history, key, mark_epoch)


def history_chart(history: dict | None = None, mark_epoch: int | None = None, height: int = 240):
    """A graph selector + the chosen graph. Returns update(history) to redraw with new data."""
    state = {"h": history or {}, "key": "loss", "chart": None}
    with ui.column().classes("w-full gap-1"):
        picker = ui.toggle({k: v[0] for k, v in CHARTS.items()}, value="loss").props(
            "dense no-caps unelevated toggle-color=amber text-color=grey-6"
        ).classes("flex-wrap font-mono text-[10px]")
        desc = ui.label("").classes("text-[#555] font-mono text-[10px] leading-snug")
        box = ui.element("div").classes("w-full")

    def build():
        box.clear()
        with box:
            state["chart"] = ui.echart(chart_options(state["h"], state["key"], mark_epoch)).classes(
                "w-full").style(f"height:{height}px")
        desc.set_text(CHARTS[state["key"]][1])

    def update(h: dict):
        state["h"] = h
        opts = chart_options(h, state["key"], mark_epoch)
        chart = state["chart"]
        if chart is None or len(opts["series"]) != len(chart.options.get("series", [])):
            build()
            return
        chart.options.clear()
        chart.options.update(opts)
        chart.update()

    def pick(e):
        state["key"] = e.value
        build()

    picker.on_value_change(pick)
    build()
    return update
