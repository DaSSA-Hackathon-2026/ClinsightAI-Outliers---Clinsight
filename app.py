import json
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# Page + Brand Styling
# -----------------------------
st.set_page_config(page_title="ClinsightAI — Review Intelligence", layout="wide")

st.markdown("""
<style>
/* Make the app feel more like a product */
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
h1, h2, h3 {letter-spacing: -0.5px;}
/* Brand bar */
.clinsight-hero {
  padding: 18px 22px;
  border-radius: 16px;
  background: linear-gradient(90deg, rgba(22,163,74,0.18), rgba(59,130,246,0.18));
  border: 1px solid rgba(255,255,255,0.10);
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.15);
  background: rgba(255,255,255,0.06);
  margin-left: 8px;
}
.card {
  padding: 14px 14px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}
.small {opacity: 0.8; font-size: 13px;}
hr {border: none; height: 1px; background: rgba(255,255,255,0.10); margin: 18px 0;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "outputs" / "clinsightai_report.json"

KW_PATH = ROOT / "outputs" / "theme_keywords.csv"
EXEC_PATH = ROOT / "outputs" / "executive_summary.txt"
DEMO_PATH = ROOT / "outputs" / "demo_script.txt"
KPI_PATH  = ROOT / "outputs" / "kpis.txt"

# -----------------------------
# Load report
# -----------------------------
if not REPORT_PATH.exists():
    st.error(f"Missing report file: {REPORT_PATH}")
    st.stop()

report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

summary = report["clinic_summary"]
theme_df = pd.DataFrame(report["theme_analysis"])
roadmap_df = pd.DataFrame(report["improvement_roadmap"])

# Merge keywords if available
if KW_PATH.exists():
    kw_df = pd.read_csv(KW_PATH)
    theme_df = theme_df.merge(kw_df, on="cluster_id", how="left")
else:
    theme_df["top_keywords"] = ""

# -----------------------------
# Sidebar: Executive Controls
# -----------------------------
st.sidebar.markdown("## Executive Controls")
st.sidebar.caption("Tune the view to match stakeholder needs.")

effort_filter = st.sidebar.selectbox(
    "Roadmap filter (Effort)",
    ["All", "Quick win", "High-effort improvement", "Reinforce & monitor"]
)

min_freq = st.sidebar.slider("Minimum theme frequency (%)", 0.0, 40.0, 0.0, 0.5)
only_negative = st.sidebar.checkbox("Show only negative drivers (rating impact < 0)", value=False)

# Systemic definition is visible + adjustable (unique touch)
st.sidebar.markdown("---")
st.sidebar.markdown("### Systemic Rule")
sys_freq = st.sidebar.slider("Systemic if frequency > (%)", 0.0, 30.0, 10.0, 0.5)
sys_impact = st.sidebar.slider("AND |impact| > ", 0.0, 2.0, 0.5, 0.05)

# Apply filters to theme_df
filtered_theme_df = theme_df.copy()
filtered_theme_df = filtered_theme_df[filtered_theme_df["frequency_percentage"] >= min_freq]
if only_negative:
    filtered_theme_df = filtered_theme_df[filtered_theme_df["rating_impact"] < 0]

# Add systemic flag column (for table + drill-down)
filtered_theme_df["systemic_flag"] = (
    (filtered_theme_df["frequency_percentage"] > sys_freq) &
    (filtered_theme_df["rating_impact"].abs() > sys_impact)
)

# -----------------------------
# Hero Header
# -----------------------------
st.markdown(f"""
<div class="clinsight-hero">
  <div style="display:flex; align-items:center; justify-content:space-between;">
    <div>
      <h1 style="margin:0;">ClinsightAI — Review Intelligence</h1>
      <div class="small">Operational themes • Rating drivers • Systemic risks • Action roadmap</div>
    </div>
    <div>
      <span class="badge">Prototype</span>
      <span class="badge">Explainable</span>
      <span class="badge">Decision Intelligence</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# KPI Cards (summary)
# -----------------------------
risks = summary.get("primary_risk_themes", [])
drivers = summary.get("primary_growth_drivers", [])

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
c1.metric("Overall Rating Mean", summary["overall_rating_mean"])
c2.metric("Primary Risks", f"{len(risks)} themes")
c3.metric("Growth Drivers", f"{len(drivers)} themes")

# Unique touch: “risk posture”
neg_share = (theme_df["rating_impact"] < 0).mean()
c4.metric("Risk Posture", f"{int(neg_share*100)}% themes negative")

with st.expander("View full risk themes & growth drivers"):
    st.markdown("**Primary Risk Themes**")
    for t in risks:
        st.write(f"• {t}")
    st.markdown("**Growth Drivers**")
    for t in drivers:
        st.write(f"• {t}")

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# Risk Snapshot Cards (Top 3)
# -----------------------------
st.subheader("Risk Snapshot")
top3 = filtered_theme_df.sort_values("risk_score", ascending=False).head(3)

cols = st.columns(3)
for i, (_, r) in enumerate(top3.iterrows()):
    with cols[i]:
        badge = "⚠ Systemic" if r["systemic_flag"] else "ℹ Not systemic"
        st.markdown(f"""
        <div class="card">
          <div style="font-size:16px; font-weight:700;">{r['theme']}</div>
          <div class="small">{badge}</div>
          <div style="margin-top:10px;"></div>
          <div><b>Risk score:</b> {round(r['risk_score'],3)}</div>
          <div><b>Freq %:</b> {r['frequency_percentage']}</div>
          <div><b>Impact:</b> {r['rating_impact']}</div>
          <div><b>Confidence:</b> {r['confidence_score']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# Risk Table
# -----------------------------
st.subheader("Risk & Impact Ranking (Explainable Table)")
show_cols = [
    "systemic_flag",
    "theme",
    "frequency_percentage",
    "average_rating",
    "rating_impact",
    "ml_importance",
    "risk_score",
    "confidence_score",
]
st.dataframe(
    filtered_theme_df[show_cols].sort_values("risk_score", ascending=False),
    use_container_width=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# Risk Quadrant Map (unique)
# -----------------------------
st.subheader("Risk Quadrant Map")
st.caption("Top-right = frequent + high risk. Bottom-left = low priority.")

plot_df = filtered_theme_df.sort_values("risk_score", ascending=False).copy()
freq_cut = float(sys_freq)      # use same threshold lines as systemic rule
risk_cut = plot_df["risk_score"].median() if len(plot_df) else 0.0

fig, ax = plt.subplots()
ax.scatter(plot_df["frequency_percentage"], plot_df["risk_score"])

# quadrant guides
ax.axvline(freq_cut, linestyle="--")
ax.axhline(risk_cut, linestyle="--")

ax.set_xlabel("Frequency (%)")
ax.set_ylabel("Risk Score")
st.pyplot(fig)

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# Theme Drill-Down
# -----------------------------
st.subheader("Theme Drill-Down (Evidence → KPI → Action)")

themes = filtered_theme_df["theme"].tolist()
if len(themes) == 0:
    st.warning("No themes match the current filters. Adjust filters in the sidebar.")
    st.stop()

selected = st.selectbox("Select a theme", themes, index=0)
row = filtered_theme_df[filtered_theme_df["theme"] == selected].iloc[0].to_dict()

left, right = st.columns([1, 1])

with left:
    st.markdown(f"### {selected}")

    is_systemic = bool(row.get("systemic_flag", False))
    if is_systemic:
        st.success("⚠ Systemic Operational Issue (recurring + high impact)")
    else:
        st.info("Lower Priority / Isolated Theme")

    st.json({
        "Frequency %": row.get("frequency_percentage"),
        "Avg rating": row.get("average_rating"),
        "Rating impact": row.get("rating_impact"),
        "ML importance": row.get("ml_importance"),
        "Risk score": row.get("risk_score"),
        "Confidence": row.get("confidence_score"),
    })

    st.markdown("**Why identified**")
    st.write(row.get("why_identified", ""))

with right:
    st.markdown("**Top keywords**")
    st.write(row.get("top_keywords", ""))

    st.markdown("**Evidence samples**")
    for i, s in enumerate(row.get("evidence_samples", [])[:3], start=1):
        st.write(f"{i}. {s}")

    st.markdown("**Suggested KPIs**")
    for k in row.get("suggested_kpis", []):
        st.write(f"• {k}")

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# Roadmap (filterable)
# -----------------------------
st.subheader("Improvement Roadmap (Prioritized)")

roadmap_show = roadmap_df[
    ["priority", "theme", "effort_bucket", "expected_rating_lift", "confidence", "recommendation"]
].sort_values("priority")

if effort_filter != "All":
    roadmap_show = roadmap_show[roadmap_show["effort_bucket"] == effort_filter]

st.dataframe(roadmap_show, use_container_width=True)

st.markdown("### KPI Checklist")
for _, r in roadmap_df.sort_values("priority").iterrows():
    with st.expander(f"Priority {int(r['priority'])}: {r['theme']}"):
        st.write(r.get("recommendation", ""))
        for k in r.get("kpis_to_track", []):
            st.checkbox(k, key=f"{r['theme']}::{k}")

st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# Export
# -----------------------------
st.subheader("Export")
st.download_button(
    label="Download JSON Report",
    data=json.dumps(report, indent=2, ensure_ascii=False),
    file_name="clinsightai_report.json",
    mime="application/json",
)

if EXEC_PATH.exists():
    st.download_button(
        "Download Executive Summary (TXT)",
        EXEC_PATH.read_text(encoding="utf-8"),
        file_name="executive_summary.txt",
        mime="text/plain",
    )

if DEMO_PATH.exists():
    st.download_button(
        "Download Demo Script (TXT)",
        DEMO_PATH.read_text(encoding="utf-8"),
        file_name="demo_script.txt",
        mime="text/plain",
    )

if KPI_PATH.exists():
    st.download_button(
        "Download KPI List (TXT)",
        KPI_PATH.read_text(encoding="utf-8"),
        file_name="kpis.txt",
        mime="text/plain",
    )

st.caption(
    "ClinsightAI turns review noise into measurable operational decisions: what to fix, why it matters, and how to track improvement."
)