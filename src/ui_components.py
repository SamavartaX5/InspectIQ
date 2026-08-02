"""Reusable, accessible presentation helpers for the public InspectIQ demo."""
from __future__ import annotations

from html import escape
from typing import Iterable, Mapping


PRIORITY_LABELS = {
    "highest_priority": "Highest priority",
    "high_priority": "High priority",
    "elevated_priority": "Standard review",
    "standard_priority": "Later review",
}


def _badge_class(value: str) -> str:
    text = value.lower()
    if text in {"pass", "healthy"}:
        return "pass"
    if text in {"warning", "warn"}:
        return "warning"
    if text in {"critical", "fail"}:
        return "critical"
    return "neutral"


def render_status_badge(st, label: str, value: str, *, kind: str | None = None) -> None:
    safe_label, safe_value = escape(str(label)), escape(str(value))
    style = _badge_class(kind or value)
    st.markdown(
        f'<span class="inspectiq-badge {style}">{safe_label}: {safe_value}</span>',
        unsafe_allow_html=True,
    )


def render_badge_row(st, badges: Iterable[tuple[str, str, str | None]]) -> None:
    for label, value, kind in badges:
        render_status_badge(st, label, value, kind=kind)


def render_page_header(st, title: str, description: str, *, eyebrow: str = "InspectIQ") -> None:
    st.markdown(f'<div class="inspectiq-eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="inspectiq-subtitle">{escape(description)}</div>', unsafe_allow_html=True)


def is_long_metric_value(value: object) -> bool:
    """Identify metric values that need a wrapping-compatible card treatment."""
    return len(str(value)) > 20


def render_kpi_row(st, items: Iterable[Mapping[str, object]]) -> None:
    values = list(items)
    columns = st.columns(len(values))
    for column, item in zip(columns, values):
        label, value = str(item["label"]), item["value"]
        if bool(item.get("long_value")) or is_long_metric_value(value):
            column.markdown(
                '<div class="inspectiq-metric-card inspectiq-metric-card--long">'
                f'<div class="inspectiq-metric-card-label">{escape(label)}</div>'
                f'<div class="inspectiq-metric-card-value">{escape(str(value))}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            column.metric(label, value, help=item.get("help"))
        if item.get("caption"):
            column.caption(str(item["caption"]))


def render_info_banner(st, text: str) -> None:
    st.markdown(f'<div class="inspectiq-callout">{escape(text)}</div>', unsafe_allow_html=True)


def render_warning_banner(st, text: str) -> None:
    st.markdown(f'<div class="inspectiq-callout warning">{escape(text)}</div>', unsafe_allow_html=True)


def render_empty_state(st, title: str, detail: str) -> None:
    st.info(f"{title}. {detail}")


def priority_label(value: str) -> str:
    return PRIORITY_LABELS.get(str(value), str(value).replace("_", " ").title())


def render_footer(st, release: str | None = None) -> None:
    suffix = f" · {escape(release)}" if release else ""
    st.markdown(
        "<div class=\"inspectiq-footer\">Advisory ranking only · Human review required · "
        "Frozen candidate ranking · Awaiting complete outcome labels · No automatic enforcement"
        f"{suffix}</div>",
        unsafe_allow_html=True,
    )
