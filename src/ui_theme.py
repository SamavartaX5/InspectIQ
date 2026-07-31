"""Scoped, dependency-free styling for the InspectIQ public dashboard."""
from __future__ import annotations


def apply_theme(st) -> None:
    """Apply static CSS only; no user content or executable script is injected."""
    st.markdown(
        """
<style>
    .stApp { background: #0b1220; }
    .block-container { max-width: 1440px; padding-top: 2.25rem; padding-bottom: 2.5rem; }
    [data-testid="stSidebar"] { background: #101a2a; border-right: 1px solid #25344b; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    h1 { font-weight: 720; }
    h2 { margin-top: 0.5rem; }
    [data-testid="stMetric"] { background: #151f31; border: 1px solid #273852; border-radius: 12px; padding: 0.85rem 1rem; }
    [data-testid="stMetricLabel"] { color: #aebed2; font-size: 0.84rem; }
    [data-testid="stMetricValue"] { color: #f4f8ff; font-size: 1.45rem; }
    [data-testid="stDataFrame"] { border: 1px solid #273852; border-radius: 10px; overflow: hidden; }
    [data-testid="stExpander"] { border: 1px solid #273852; border-radius: 10px; background: #111b2b; }
    [data-testid="stDownloadButton"] > button, [data-testid="stButton"] > button {
        border-radius: 8px; border: 1px solid #3e7190; font-weight: 600;
    }
    [data-testid="stDownloadButton"] > button:hover, [data-testid="stButton"] > button:hover {
        border-color: #63cdf4; background: #17324a;
    }
    [data-baseweb="select"] > div { border-radius: 8px; }
    hr { border-color: #273852; margin: 1.35rem 0; }
    .inspectiq-eyebrow { color: #75d6fa; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    .inspectiq-subtitle { color: #b7c6d8; font-size: 1.06rem; max-width: 760px; margin: 0.35rem 0 0.85rem; }
    .inspectiq-badge { display: inline-block; margin: 0.15rem 0.35rem 0.15rem 0; padding: 0.24rem 0.55rem; border-radius: 999px; font-size: 0.78rem; font-weight: 700; border: 1px solid #3c536f; background: #16243a; color: #d9e9fa; }
    .inspectiq-badge.warning { border-color: #9b751f; background: #322816; color: #ffd98a; }
    .inspectiq-badge.pass, .inspectiq-badge.healthy { border-color: #2d8058; background: #143322; color: #9de3bd; }
    .inspectiq-badge.critical { border-color: #ad4f57; background: #3a1c22; color: #ffb5bb; }
    .inspectiq-badge.neutral { border-color: #3c536f; background: #16243a; color: #d9e9fa; }
    .inspectiq-callout { border-left: 3px solid #4fc3f7; background: #132338; padding: 0.8rem 0.95rem; border-radius: 0 9px 9px 0; color: #dbe8f6; margin: 0.7rem 0 1rem; }
    .inspectiq-callout.warning { border-left-color: #e8b84d; background: #312714; }
    .inspectiq-footer { color: #91a3ba; border-top: 1px solid #273852; margin-top: 2rem; padding-top: 1rem; font-size: 0.84rem; }
    .inspectiq-sidebar-card { border: 1px solid #31445f; border-radius: 10px; padding: 0.75rem; background: #152238; color: #dce8f5; font-size: 0.87rem; }
    @media (max-width: 900px) { .block-container { padding-left: 1rem; padding-right: 1rem; } [data-testid="stMetricValue"] { font-size: 1.22rem; } }
</style>
        """,
        unsafe_allow_html=True,
    )
