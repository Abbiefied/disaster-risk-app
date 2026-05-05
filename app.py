"""
app.py - Disaster Risk Intelligence System
Group 5 | SheCodeAfrica Data Science Bootcamp 2026
Topic  : Predicting the Occurrence Rate of Natural Disasters in Asia
Model  : Logistic Regression on EM-DAT country-year panel data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import textwrap

from config import *
from utils import (
    predict_risk,
    build_next_year_input,
    get_feature_importance,
    get_risk_drivers,
    risk_label,
)

#Page config
st.set_page_config(
    page_title="Disaster Risk Intelligence | Asia",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#Custom CSS
st.markdown("""
<style>
/*Google Fonts*/
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/*Background*/
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}

/*Header*/
[data-testid="stHeader"],
[data-testid="stToolbar"],
header[data-testid="stHeader"] {
    background: #0f1117 !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}

/* Header text, buttons and icons */
[data-testid="stHeader"] button,
[data-testid="stHeader"] a,
[data-testid="stHeader"] span,
[data-testid="stHeader"] p,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] span,
[data-testid="stDecoration"] {
    color: #c8d0e8 !important;
    fill: #c8d0e8 !important;
}

/* Deploy button specifically */
[data-testid="stAppDeployButton"] button,
[data-testid="stAppDeployButton"] span {
    color: #c8d0e8 !important;
    border-color: rgba(255,255,255,0.2) !important;
}

/* Sidebar collapse toggle arrow */
[data-testid="stSidebarCollapsedControl"] button {
    color: #c8d0e8 !important;
}

/* Widget labels (selectbox, multiselect, slider, etc.) */
label, .stSelectbox label, .stMultiSelect label,
div[data-testid="stWidgetLabel"] > div,
div[data-testid="stWidgetLabel"] p { color: #8892b0 !important; }
 
/* Selectbox - selected value text inside the box */
div[data-testid="stSelectbox"] > div > div > div,
div[data-testid="stSelectbox"] > div[data-baseweb="select"] span {
    color: #e8eaf0 !important;
    background: #1c2238 !important;
}
 
/* Selectbox - dropdown popup list items */
ul[data-testid="stSelectboxVirtualDropdown"] li,
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li,
div[data-baseweb="menu"] li span,
div[role="listbox"] li,
div[role="option"]  { color: #e8eaf0 !important; background: #1c2238 !important; }
div[data-baseweb="menu"]             { background: #1c2238 !important; }
div[data-baseweb="menu"] li:hover,
div[role="option"]:hover             { background: #252d47 !important; }
 
/* Multiselect - tags and dropdown */
div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(91,127,245,0.2) !important;
    color: #8aadff !important;
}
div[data-testid="stMultiSelect"] input { color: #e8eaf0 !important; }
 
/* Input box background */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div:first-child {
    background: #1c2238 !important;
    border-color: rgba(255,255,255,0.1) !important;
}
 
/* Expander header text */
details summary p,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span { color: #c8d0e8 !important; }
div[data-testid="stExpander"] {
    background: #161b2e !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] > div { color: #8892b0 !important; }
 
/* Caption / small text */
div[data-testid="stCaptionContainer"] p,
small, .caption { color: #5c6a8a !important; }

/* Tabs */
div[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #5c6a8a !important;
    background: transparent !important;
}
div[data-testid="stTabs"] button:hover        { color: #c8d0e8 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #8aadff !important;
    border-bottom-color: #5b7ff5 !important;
}
div[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}
 
/* Subheader */
h2, h3 { color: #c8d0e8 !important; }
 
/* Horizontal rule */
hr { border-color: rgba(255,255,255,0.07) !important; }
 
/* Dataframe / table */
div[data-testid="stDataFrame"]           { border-radius: 10px; overflow: hidden; }
div[data-testid="stDataFrame"] th        { background: #1c2238 !important; color: #8892b0 !important; }
div[data-testid="stDataFrame"] td        { color: #e8eaf0 !important; background: #161b2e !important; }
div[data-testid="stDataFrame"] tr:hover td { background: #1e2440 !important; }
 
/* Download button */
div[data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1px solid rgba(91,127,245,0.4) !important;
    color: #8aadff !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
div[data-testid="stDownloadButton"] button:hover {
    background: rgba(91,127,245,0.12) !important;
    border-color: #5b7ff5 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #12192e !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label { color: #8892b0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #c8d0e8 !important; }
 
/* Scrollbar */
::-webkit-scrollbar             { width: 6px; height: 6px; }
::-webkit-scrollbar-track       { background: #0f1117; }
::-webkit-scrollbar-thumb       { background: #2a3050; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3d5af1; }

/*Header banner*/
.hero {
    background: linear-gradient(135deg, #1a1f35 0%, #12192e 60%, #0f1117 100%);
    border: 1px solid rgba(100,140,255,0.15);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,130,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: 0.2rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8892b0;
    margin: 0;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,130,255,0.15);
    border: 1px solid rgba(99,130,255,0.3);
    color: #8aadff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* Stat pill (sidebar) */
.stat-pill {
    background: #1c2238;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stat-pill-label { font-size: 0.78rem; color: #5c6a8a !important; }
.stat-pill-value { font-size: 1.1rem; font-weight: 600; color: #c8d0e8 !important; }

/*Card*/
.card {
    background: #161b2e;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5c6a8a;
    margin-bottom: 0.8rem;
}

/*Risk badge*/
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.4rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.05rem;
    margin-top: 0.3rem;
}

/*Probability dial label*/
.prob-number {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    font-weight: 400;
    line-height: 1;
    margin: 0;
}

/*Section heading*/
.section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #c8d0e8;
    margin: 0 0 1rem 0;
}

/*Driver row*/
.driver-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.88rem;
}
.driver-row:last-child { border-bottom: none; }
.driver-up   { color: #e17070; font-weight: 600; }
.driver-down { color: #6ec98f; font-weight: 600; }

/* TEAM CARD */
.team-card {
    background: linear-gradient(145deg, #161b2e, #12192e);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.6rem 1.4rem;
    text-align: center;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.team-card:hover {
    transform: translateY(-6px) scale(1.01);
    border-color: rgba(91,127,245,0.35);
    box-shadow: 0 12px 30px rgba(61, 90, 241, 0.15);
}

/* soft glow */
.team-card::before {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 140px;
    height: 140px;
    background: radial-gradient(circle, rgba(91,127,245,0.18), transparent 70%);
    border-radius: 50%;
}

/* avatar */
.team-avatar {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3d5af1, #5b7ff5);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1rem;
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #ffffff;
}

/* name */
.team-name {
    font-size: 0.98rem;
    font-weight: 600;
    color: #e6ebff;
    margin-bottom: 0.25rem;
}

/* role / id */
.team-id {
    font-size: 0.72rem;
    color: #6c7aa5;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}

/* linkedin button */
.team-link {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #8aadff;
    text-decoration: none;
    background: rgba(91,127,245,0.12);
    border: 1px solid rgba(91,127,245,0.3);
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    transition: all 0.2s ease;
}

.team-link:hover {
    background: rgba(91,127,245,0.25);
    border-color: #5b7ff5;
}

/*Streamlit component overrides*/
div[data-testid="stSelectbox"] label,
div[data-testid="stMetricLabel"]   { color: #8892b0 !important; }
div[data-testid="stMetricValue"]   { color: #ffffff !important; font-size: 1.8rem !important; }

button[kind="primary"] {
    background: linear-gradient(135deg, #3d5af1, #5b7ff5) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 0.55rem 1.4rem !important;
}

div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

#Load data 
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

cy = load_data()

#Sidebar 
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem">
        <div style="font-size:2.2rem">🌏</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;
                    color:#c8d0e8;margin-top:0.4rem">
            Disaster Risk<br>Intelligence
        </div>
        <div style="font-size:0.7rem;color:#5c6a8a;margin-top:0.3rem;
                    letter-spacing:1px;text-transform:uppercase">
            Group 5 · SCA Bootcamp 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem;letter-spacing:1.5px;'
                'text-transform:uppercase;color:#5c6a8a;margin-bottom:0.8rem">'
                'Dataset Summary</p>', unsafe_allow_html=True)

    n_countries = cy["Country"].nunique()
    n_years     = cy["Start_Year"].nunique()
    year_range  = f"{int(cy['Start_Year'].min())}–{int(cy['Start_Year'].max())}"
    total_events= int(cy["event_count"].sum())
    high_pct    = f"{(cy['High_Occurrence'].mean()*100):.0f}%" if "High_Occurrence" in cy.columns else "-"

    for label, value in [
        ("Countries", str(n_countries)),
        ("Years covered", year_range),
        ("Total records", f"{len(cy):,}"),
        ("Total events", f"{total_events:,}"),
        ("High-risk rate", high_pct),
    ]:
        st.markdown(
            f'<div class="stat-pill">'
            f'<span class="stat-pill-label">{label}</span>'
            f'<span class="stat-pill-value">{value}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem;letter-spacing:1.5px;'
                'text-transform:uppercase;color:#5c6a8a;margin-bottom:0.8rem">'
                'Model</p>', unsafe_allow_html=True)
    for chip in ["Logistic Regression", "L2 regularisation",
                 "class_weight=balanced", "5-Fold CV", "Temporal Split"]:
        st.markdown(f'<span class="insight-chip">{chip}</span>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.72rem;color:#3d4a6a;text-align:center">'
        'EM-DAT · CRED · UCLouvain<br>Data: Asia 2001–2024</p>',
        unsafe_allow_html=True
    )

#Hero banner 
# Compute live headline stats for the hero
most_active_country = cy.groupby("Country")["event_count"].sum().idxmax()
most_active_year    = cy.groupby("Start_Year")["event_count"].sum().idxmax()
avg_events_yr       = cy.groupby("Start_Year")["event_count"].sum().mean()

st.markdown(f"""
<div class="hero">
    <div class="hero-tag">🌏 Group 5 · SheCodeAfrica Data Science Bootcamp 2026</div>
    <p class="hero-title">Disaster Risk Intelligence System</p>
    <p class="hero-sub">
        Predicting the occurrence rate of natural disasters across Asia &nbsp;·&nbsp;
        Logistic Regression on EM-DAT panel data (2001–2024)
    </p>
    <div style="display:flex;gap:2rem;margin-top:1.8rem;flex-wrap:wrap">
        <div>
            <div style="font-size:0.68rem;letter-spacing:1.2px;text-transform:uppercase;
                        color:#3d5a8a;margin-bottom:0.2rem">Most active country</div>
            <div style="font-size:1.15rem;font-weight:600;color:#8aadff">
                {most_active_country}
            </div>
        </div>
        <div>
            <div style="font-size:0.68rem;letter-spacing:1.2px;text-transform:uppercase;
                        color:#3d5a8a;margin-bottom:0.2rem">Peak year</div>
            <div style="font-size:1.15rem;font-weight:600;color:#8aadff">
                {most_active_year}
            </div>
        </div>
        <div>
            <div style="font-size:0.68rem;letter-spacing:1.2px;text-transform:uppercase;
                        color:#3d5a8a;margin-bottom:0.2rem">Avg events / year</div>
            <div style="font-size:1.15rem;font-weight:600;color:#8aadff">
                {avg_events_yr:.0f}
            </div>
        </div>
        <div>
            <div style="font-size:0.68rem;letter-spacing:1.2px;text-transform:uppercase;
                        color:#3d5a8a;margin-bottom:0.2rem">Countries tracked</div>
            <div style="font-size:1.15rem;font-weight:600;color:#8aadff">
                {n_countries}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


#Tabs 
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict Risk",
    "🏆 Country Rankings",
    "📈 Trend Analysis",
    "🧠 Model Insights",
    "👥 Our Team",
])


# ---------------------------------------------------
# TAB 1 - PREDICTION
# ---------------------------------------------------
with tab1:

    st.markdown('<p class="section-heading">Country Risk Prediction</p>',
                unsafe_allow_html=True)

    col_sel1, col_sel2 = st.columns([1, 1])

    with col_sel1:
        country = st.selectbox("🌐 Country", sorted(cy["Country"].unique()))

    with col_sel2:
        available_years = sorted(cy[cy["Country"] == country]["Start_Year"].unique())
        year = st.selectbox("📅 Year", available_years)

    selected_row = cy[
        (cy["Country"] == country) &
        (cy["Start_Year"] == year)
    ]

    if selected_row.empty:
        st.warning("No data found for this country-year combination.")
        st.stop()

    st.markdown("---")

    #Side-by-side current / next year predictions
    col_curr, col_next = st.columns(2)

    input_df = selected_row.drop(
        columns=["Country", "High_Occurrence", "event_count",
                 "total_deaths", "total_affected", "total_damage"],
        errors="ignore"
    )
    pred_curr, prob_curr = predict_risk(input_df)
    label_curr, colour_curr = risk_label(prob_curr)

    next_row   = build_next_year_input(selected_row)
    next_input = next_row.drop(
        columns=["Country", "High_Occurrence", "event_count",
                 "total_deaths", "total_affected", "total_damage"],
        errors="ignore"
    )
    pred_next, prob_next = predict_risk(next_input)
    label_next, colour_next = risk_label(prob_next)

    def _badge(pred, label, colour):
        icon = "⚠" if pred else "✔"
        bg   = "rgba(231,76,60,0.15)" if pred else "rgba(46,204,113,0.12)"
        return (
            f'<div style="display:inline-flex;align-items:center;gap:0.5rem;'
            f'padding:0.6rem 1.4rem;border-radius:50px;font-weight:600;'
            f'font-size:1.05rem;margin-top:0.3rem;background:{bg};'
            f'border:1px solid {colour};color:{colour}">'
            f'{icon} {label} Risk</div>'
        )

    def _progress(prob, colour):
        pct = int(prob * 100)
        return (
            f'<div style="height:8px;border-radius:99px;'
            f'background:rgba(255,255,255,0.07);margin-top:1rem">'
            f'<div style="height:8px;border-radius:99px;width:{pct}%;'
            f'background:{colour}"></div></div>'
        )

    with col_curr:
        st.markdown(
            f'<div style="background:#161b2e;border:1px solid rgba(255,255,255,0.07);'
            f'border-radius:12px;padding:1.5rem 1.8rem;margin-bottom:1.2rem">'
            f'<p style="font-size:0.72rem;font-weight:600;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:#5c6a8a;margin-bottom:0.8rem">Current Year</p>'
            f'<p style="font-family:\'DM Serif Display\',serif;font-size:3.2rem;'
            f'font-weight:400;line-height:1;margin:0;color:{colour_curr}">{prob_curr:.1%}</p>'
            f'{_badge(pred_curr, label_curr, colour_curr)}'
            f'<p style="margin-top:0.8rem;font-weight:600;color:#c8d0e8">{country} — {year}</p>'
            f'{_progress(prob_curr, colour_curr)}'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_next:
        st.markdown(
            f'<div style="background:#161b2e;border:1px solid rgba(255,255,255,0.07);'
            f'border-radius:12px;padding:1.5rem 1.8rem;margin-bottom:1.2rem">'
            f'<p style="font-size:0.72rem;font-weight:600;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:#5c6a8a;margin-bottom:0.8rem">Next Year Forecast</p>'
            f'<p style="font-family:\'DM Serif Display\',serif;font-size:3.2rem;'
            f'font-weight:400;line-height:1;margin:0;color:{colour_next}">{prob_next:.1%}</p>'
            f'{_badge(pred_next, label_next, colour_next)}'
            f'<p style="margin-top:0.8rem;font-weight:600;color:#c8d0e8">{country} — {year + 1} (projected)</p>'
            f'{_progress(prob_next, colour_next)}'
            f'</div>',
            unsafe_allow_html=True
        )

    #Risk delta callout 
    delta      = prob_next - prob_curr
    delta_icon = "📈" if delta > 0.02 else "📉" if delta < -0.02 else "➡️"
    delta_text = (
        f"Risk is projected to **increase by {abs(delta):.1%}**"
        if delta > 0.02 else
        f"Risk is projected to **decrease by {abs(delta):.1%}**"
        if delta < -0.02 else
        "Risk is projected to **remain stable** year-on-year."
    )
    st.info(f"{delta_icon} {delta_text}")

    #Risk drivers panel 
    st.markdown("---")
    st.markdown('<p class="section-heading">What is driving the risk?</p>',
                unsafe_allow_html=True)
    st.caption(
        "Each row shows how much a feature pushes the model's prediction "
        "toward High Risk (red) or Low Risk (green), based on its scaled "
        "value multiplied by the logistic regression coefficient."
    )

    drivers = get_risk_drivers(input_df, top_n=6)

    for _, row in drivers.iterrows():
        is_up   = row["Contribution"] > 0
        colour  = "#e17070" if is_up else "#6ec98f"
        arrow   = "&#9650;" if is_up else "&#9660;"
        bar_pct = min(abs(row["Contribution"]) / drivers["Contribution"].abs().max(), 1.0)
        bar_w   = int(bar_pct * 180)

        st.markdown(f"""
        <div class="driver-row">
            <span style="width:220px;overflow:hidden;text-overflow:ellipsis;
                         white-space:nowrap;color:#c8d0e8">
                {row['Feature']}
            </span>
            <div style="flex:1;margin:0 1rem">
                <div style="height:6px;border-radius:3px;background:rgba(255,255,255,0.06)">
                    <div style="height:6px;border-radius:3px;width:{bar_w}px;
                                background:{colour};transition:width 0.4s ease"></div>
                </div>
            </div>
            <span style="color:{colour};font-weight:600;min-width:80px;text-align:right">
                {arrow} {abs(row['Contribution']):.3f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    #Country historical profile 
    st.markdown("---")
    st.markdown(f'<p class="section-heading">Historical Profile - {country}</p>',
                unsafe_allow_html=True)
    st.caption("Annual event count for the selected country across all available years."
               " The selected year is highlighted.")

    country_history = cy[cy["Country"] == country].sort_values("Start_Year")

    fig_hist = go.Figure()

    # Area fill
    fig_hist.add_trace(go.Scatter(
        x=country_history["Start_Year"],
        y=country_history["event_count"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(91,127,245,0.10)",
        line=dict(color="#5b7ff5", width=2),
        name="Event Count",
        hovertemplate="<b>%{x}</b><br>Events: %{y}<extra></extra>",
    ))

    # Highlight selected year
    sel = country_history[country_history["Start_Year"] == year]
    if not sel.empty:
        fig_hist.add_trace(go.Scatter(
            x=sel["Start_Year"],
            y=sel["event_count"],
            mode="markers",
            marker=dict(size=12, color=colour_curr,
                        line=dict(width=2, color="#ffffff")),
            name=f"Selected ({year})",
            hovertemplate=f"<b>{year}</b><br>Events: %{{y}}<extra></extra>",
        ))

    fig_hist.update_layout(
        plot_bgcolor="#161b2e",
        paper_bgcolor="#161b2e",
        font_color="#c8d0e8",
        height=260,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Year"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Events"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892b0")),
        showlegend=True,
    )
    st.plotly_chart(fig_hist, width='stretch')

# ---------------------------------------------------
# TAB 2 - COUNTRY RANKINGS
# ---------------------------------------------------
with tab2:

    st.markdown('<p class="section-heading">Country Risk Rankings</p>',
                unsafe_allow_html=True)

    selected_year = st.selectbox(
        "Select year for cross-country comparison",
        sorted(cy["Start_Year"].unique(), reverse=True),
        key="rank_year"
    )

    year_data = cy[cy["Start_Year"] == selected_year].copy()

    if year_data.empty:
        st.warning("No data for selected year.")
    else:
        # Vectorised: prepare all rows at once instead of looping
        drop_cols  = ["Country", "High_Occurrence", "event_count",
                      "total_deaths", "total_affected", "total_damage"]
        input_bulk = year_data.drop(columns=drop_cols, errors="ignore")

        from utils import prepare_input, model as lr_model
        X_bulk  = prepare_input(input_bulk)
        probs   = lr_model.predict_proba(X_bulk)[:, 1]
        preds   = lr_model.predict(X_bulk)

        ranking_df = pd.DataFrame({
            "Country"         : year_data["Country"].values,
            "Risk Probability": probs,
            "Risk Level"      : [risk_label(p)[0] for p in probs],
            "Prediction"      : ["High" if p else "Low" for p in preds],
        }).sort_values("Risk Probability", ascending=False).reset_index(drop=True)

        ranking_df.index += 1   # 1-based rank

        col_tbl, col_chart = st.columns([1, 1.4])

        with col_tbl:
            st.markdown(f"**{selected_year} - All countries ranked by risk probability**")
            st.dataframe(
                ranking_df.style.background_gradient(
                    subset=["Risk Probability"], cmap="RdYlGn_r"
                ).format({"Risk Probability": "{:.1%}"}),
                width='stretch',
                height=500,
            )

        with col_chart:
            top15 = ranking_df.head(15).sort_values("Risk Probability")
            colours = [risk_label(p)[1] for p in top15["Risk Probability"]]

            fig = go.Figure(go.Bar(
                x=top15["Risk Probability"],
                y=top15["Country"],
                orientation="h",
                marker_color=colours,
                text=[f"{p:.1%}" for p in top15["Risk Probability"]],
                textposition="outside",
            ))
            fig.update_layout(
                title=dict(
                    text=f"Top 15 Countries by Risk Probability ({selected_year})", 
                    font=dict(color='#c8d0e8')
                ),
                xaxis_title="Risk Probability",
                plot_bgcolor="#161b2e",
                paper_bgcolor="#161b2e",
                font_color="#c57424",
                height=520, 
                margin=dict(l=10, r=60, t=50, b=10),
                xaxis=dict(range=[0, 1.1], gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig, width='stretch')

        #Asia choropleth map 
        st.markdown("---")
        st.markdown(
            f'<p class="section-heading">Asia Risk Map - {selected_year}</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Colour intensity represents the model's predicted risk probability "
            "for each country. Countries in the dataset but with no data for the "
            "selected year appear grey. Hover over any country for details."
        )

        # Merge ISO codes from cy into ranking_df so the map has location codes
        iso_lookup = (
            cy[["Country", "ISO"]]
            .drop_duplicates("Country")
            .set_index("Country")["ISO"]
            .to_dict()
        )
        ranking_df["ISO"] = ranking_df["Country"].map(iso_lookup)

        # Risk level order for the hover label
        ranking_df["Risk %"] = (ranking_df["Risk Probability"] * 100).round(1)

        fig_map = go.Figure(go.Choropleth(
            locations      = ranking_df["ISO"],
            z              = ranking_df["Risk Probability"],
            text           = ranking_df["Country"],
            customdata     = np.stack([
                ranking_df["Risk %"],
                ranking_df["Risk Level"],
                ranking_df["Prediction"],
            ], axis=-1),
            hovertemplate  = (
                "<b>%{text}</b><br>"
                "Risk Probability : %{customdata[0]:.1f}%<br>"
                "Risk Level       : %{customdata[1]}<br>"
                "Prediction       : %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            colorscale     = [
                [0.00, "#1a3a5c"],   # deep blue  - very low risk
                [0.25, "#2e6da4"],   # mid blue   - low risk
                [0.50, "#f1c40f"],   # amber      - moderate risk
                [0.75, "#e67e22"],   # orange     - high risk
                [1.00, "#c0392b"],   # deep red   - very high risk
            ],
            zmin           = 0,
            zmax           = 1,
            marker_line_color = "rgba(255,255,255,0.15)",
            marker_line_width = 0.5,
            colorbar = dict(
                title      = dict(text="Risk Probability", font=dict(color="#c8d0e8")),
                tickformat = ".0%",
                tickfont   = dict(color="#c8d0e8"),
                bgcolor    = "rgba(22,27,46,0.8)",
                bordercolor= "rgba(255,255,255,0.1)",
                borderwidth= 1,
                thickness  = 16,
                len        = 0.75,
                tickvals   = [0, 0.25, 0.50, 0.75, 1.0],
                ticktext   = ["0% Low", "25%", "50% Moderate", "75%", "100% High"],
            ),
        ))

        fig_map.update_layout(
            geo = dict(
                scope           = "asia",
                showframe       = False,
                showcoastlines  = True,
                coastlinecolor  = "rgba(255,255,255,0.12)",
                showland        = True,
                landcolor       = "#1a1f35",
                showocean       = True,
                oceancolor      = "#0f1117",
                showlakes       = True,
                lakecolor       = "#0f1117",
                showcountries   = True,
                countrycolor    = "rgba(255,255,255,0.08)",
                bgcolor         = "#0f1117",
                projection_type = "natural earth",
            ),
            paper_bgcolor = "#0f1117",
            font_color    = "#c8d0e8",
            title         = dict(
                text      = f"Predicted Disaster Risk by Country - {selected_year}",
                font      = dict(size=16, color="#c8d0e8"),
                x         = 0.5,
                xanchor   = "center",
            ),
            margin        = dict(l=0, r=0, t=50, b=0),
            height        = 520,
        )

        st.plotly_chart(fig_map, width='stretch')

        #Risk level legend 
        legend_items = [
            ("#1a3a5c", "#2e6da4", "Low (0–40%)"),
            ("#f1c40f", "#f1c40f", "Moderate (40–55%)"),
            ("#e67e22", "#e67e22", "High (55–75%)"),
            ("#c0392b", "#c0392b", "Very High (75–100%)"),
        ]
        legend_html = '<div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:0.5rem">'
        for bg, border, label in legend_items:
            legend_html += (
                f'<div style="display:flex;align-items:center;gap:0.5rem;'
                f'font-size:0.82rem;color:#8892b0">'
                f'<div style="width:14px;height:14px;border-radius:3px;'
                f'background:{bg};border:1px solid {border}"></div>{label}</div>'
            )
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

        #Download ranking data 
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = ranking_df.drop(columns=["ISO", "Risk %"], errors="ignore")
        csv_data["Risk Probability"] = csv_data["Risk Probability"].round(4)
        st.download_button(
            label="⬇ Download Rankings as CSV",
            data=csv_data.to_csv(index=False).encode("utf-8"),
            file_name=f"asia_disaster_risk_rankings_{selected_year}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------
# TAB 3 - TRENDS
# ---------------------------------------------------
with tab3:

    st.markdown('<p class="section-heading">Temporal Trend Analysis</p>',
                unsafe_allow_html=True)

    #Overall Asia trend 
    yearly = cy.groupby("Start_Year")["event_count"].sum().reset_index()

    fig_trend = px.area(
        yearly,
        x="Start_Year",
        y="event_count",
        title="Total Disaster Events Across Asia (2001–2024)", 
        labels={"Start_Year": "Year", "event_count": "Event Count"},
        color_discrete_sequence=["#5b7ff5"],
    )
    fig_trend.update_traces(
        line_width=2.5,
        fillcolor="rgba(91,127,245,0.15)"
    )
    fig_trend.update_layout(
        title_font=dict(color='#c8d0e8'),
        plot_bgcolor="#161b2e",
        paper_bgcolor="#161b2e",
        font_color="#c8d0e8",
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_trend, width='stretch')

    #Country comparison 
    st.markdown("---")
    st.subheader("Country Comparison")

    all_countries    = sorted(cy["Country"].unique())
    default_top5     = list(cy["Country"].value_counts().head(5).index)
    selected_compare = st.multiselect(
        "Select countries to compare",
        all_countries,
        default=default_top5,
        key="trend_countries"
    )

    if selected_compare:
        subset = cy[cy["Country"].isin(selected_compare)]
        fig2 = px.line(
            subset,
            x="Start_Year",
            y="event_count",
            color="Country",
            title="Annual Disaster Count - Country Comparison", 
            labels={"Start_Year": "Year", "event_count": "Event Count"},
            markers=True,
        )
        fig2.update_layout(
            title_font=dict(color='#c8d0e8'),
            plot_bgcolor="#161b2e",
            paper_bgcolor="#161b2e",
            font_color="#c8d0e8",
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig2, width='stretch')

    #Heatmap: country × year 
    st.markdown("---")
    st.subheader("Risk Heatmap - Top 15 Countries × Year")

    top15_c   = list(cy["Country"].value_counts().head(15).index)
    heat_data = (
        cy[cy["Country"].isin(top15_c)]
          .pivot_table(index="Country", columns="Start_Year",
                       values="event_count", aggfunc="sum")
          .fillna(0)
    )

    fig_heat = px.imshow(
        heat_data,
        color_continuous_scale="Blues",
        title="Disaster Event Count Heatmap",
        labels=dict(x="Year", y="Country", color="Events"),
        aspect="auto",
    )
    fig_heat.update_layout(
        title_font=dict(color='#c8d0e8'),
        plot_bgcolor="#161b2e",
        paper_bgcolor="#161b2e",
        font_color="#c8d0e8",
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar=dict(tickfont=dict(color="#c8d0e8")),
    )
    st.plotly_chart(fig_heat, width='stretch')


# ---------------------------------------------------
# TAB 4 - MODEL INSIGHTS
# ---------------------------------------------------
with tab4:

    st.markdown('<p class="section-heading">Model Insights</p>',
                unsafe_allow_html=True)

    #Model card 
    st.markdown("""
    <div class="card">
        <p class="card-title">Model Card</p>
        <table style="width:100%;font-size:0.9rem;border-collapse:collapse">
            <tr><td style="color:#5c6a8a;padding:0.4rem 0;width:200px">Algorithm</td>
                <td style="color:#c8d0e8">Logistic Regression (L2 regularisation)</td></tr>
            <tr><td style="color:#5c6a8a;padding:0.4rem 0">Prediction unit</td>
                <td style="color:#c8d0e8">Country-Year</td></tr>
            <tr><td style="color:#5c6a8a;padding:0.4rem 0">Target variable</td>
                <td style="color:#c8d0e8">High Occurrence (≥ country median annual events)</td></tr>
            <tr><td style="color:#5c6a8a;padding:0.4rem 0">Data source</td>
                <td style="color:#c8d0e8">EM-DAT (CRED), filtered to Asia 2001–2024</td></tr>
            <tr><td style="color:#5c6a8a;padding:0.4rem 0">Validation</td>
                <td style="color:#c8d0e8">Stratified 5-Fold CV + Temporal Split (2001–2019 train / 2020–2024 test)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    #Feature importance chart 
    st.markdown("---")
    st.subheader("Feature Importance - Logistic Regression Coefficients")
    st.caption(
        "Features are standardised before training, so coefficients are "
        "directly comparable. Positive (blue) = raises log-odds of High Occurrence. "
        "Negative (red) = lowers log-odds."
    )

    imp_df = get_feature_importance().head(20)

    fig_imp = go.Figure(go.Bar(
        x=imp_df["Coefficient"],
        y=imp_df["Feature"],
        orientation="h",
        marker_color=[
            "#5b7ff5" if c > 0 else "#e17070"
            for c in imp_df["Coefficient"]
        ],
        text=[f"{c:+.3f}" for c in imp_df["Coefficient"]],
        textposition="outside",
    ))
    fig_imp.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig_imp.update_layout(
        title=dict(
            text="Top 20 Feature Coefficients (by absolute magnitude)", 
            font=dict(color='#c8d0e8')
            ),
        plot_bgcolor="#161b2e",
        paper_bgcolor="#161b2e",
        font_color="#c8d0e8",
        height=560,
        margin=dict(l=10, r=80, t=50, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", autorange="reversed"),
    )
    st.plotly_chart(fig_imp, width='stretch')

    #Methodology notes for judges 
    st.markdown("---")
    st.subheader("Methodological Notes")

    with st.expander("Target variable design - why a country-specific median?"):
        st.markdown("""
        A single global threshold would be unfair: a "high" year for Bhutan
        (low disaster baseline) should not be judged by China's scale.
        We compute each country's own median annual event count and classify
        years at or above that median as **High Occurrence (1)**.
        This ensures every country is measured against its own historical baseline.
        """)

    with st.expander("Why Logistic Regression and not a more complex model?"):
        st.markdown("""
        Interpretability is a first-order requirement for this project.
        Logistic regression produces signed, magnitude-ranked coefficients that
        directly answer *what drives risk* - a question expert judges and
        policymakers can act on. More complex models (Random Forest, XGBoost)
        would likely improve AUC-ROC slightly but at the cost of interpretability.
        We validated this choice by verifying that the learning curve shows no
        significant underfitting signal.
        """)

    with st.expander("Temporal leakage prevention - how are lag features constructed?"):
        st.markdown("""
        All lag features (`prev_year_count`, `log_prev_total_deaths`, etc.)
        use data from year **t-1** to predict year **t**. The scaler was
        fitted exclusively on the training set and applied to the test set
        without refitting. `event_count` (the basis for the target variable)
        was explicitly excluded from the feature set.

        **Next-year forecasting** rolls the current year's actuals into the
        lag columns before predicting, so the model always receives the correct
        temporal input - not the same values repeated.
        """)

    with st.expander("Class imbalance - why does the positive rate exceed 50%?"):
        st.markdown("""
        A pure median split produces ~50/50 classes in theory.
        When a country has many years tied at exactly its median count,
        the `≥ median` rule classifies all ties as High Occurrence,
        pushing the positive rate above 50%.
        We account for this with `class_weight='balanced'` in the
        Logistic Regression, which re-weights the loss function to treat
        both classes equally during training.
        """)

# ---------------------------------------------------
# TAB 5 - OUR TEAM
# ---------------------------------------------------
with tab5:
 
    st.markdown('<p class="section-heading">Our Team</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8892b0;font-size:0.92rem;margin-bottom:2rem">'
        'SheCodeAfrica · Cohort 3 Data Science Bootcamp 2026 · Group 5</p>',
        unsafe_allow_html=True
    )
 
    # LinkedIn SVG icon
    LI_SVG = (
        '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">'
        '<path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037'
        '-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046'
        'c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286z'
        'M5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063'
        ' 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065z'
        'M6.914 20.452H3.76V9h3.154v11.452z'
        'M22.225 0H1.771C.792 0 0 .774 0 1.729v20.542'
        'C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729'
        'C24 .774 23.2 0 22.222 0h.003z"/></svg>'
    )

    #Github SVG icon
    GH_SVG = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">'
    '<path d="M12 .5C5.73.5.98 5.24.98 11.52c0 4.88 3.16 9.02 7.55 10.48'
    ' .55.1.75-.24.75-.53 0-.26-.01-1.14-.02-2.07-3.07.67-3.72-1.48'
    ' -3.72-1.48-.5-1.28-1.22-1.62-1.22-1.62-1-.68.08-.67.08-.67'
    ' 1.1.08 1.68 1.14 1.68 1.14.98 1.68 2.57 1.2 3.2.92.1-.71.38-1.2'
    ' .7-1.48-2.45-.28-5.02-1.22-5.02-5.43 0-1.2.43-2.18 1.13-2.95'
    ' -.11-.28-.49-1.41.11-2.94 0 0 .92-.29 3.02 1.13a10.5 10.5 0 0 1'
    ' 5.5 0c2.1-1.42 3.02-1.13 3.02-1.13.6 1.53.22 2.66.11 2.94'
    ' .7.77 1.13 1.75 1.13 2.95 0 4.22-2.58 5.14-5.04 5.42'
    ' .39.34.74 1.01.74 2.04 0 1.47-.01 2.65-.01 3.01 0 .29.2.64.76.53'
    ' 4.38-1.46 7.54-5.6 7.54-10.48C23.02 5.24 18.27.5 12 .5z"/>'
    '</svg>'
    )
    
    #Team members 
    team_members = [
        ("Azeezat Kareem",             "SCA/APC3/DS/011", "https://linkedin.com/in/azeezat-kareem", "https://github.com/Abbiefied/disaster-risk-app"),
        ("Ganiyat Adekunle",           "SCA/APC3/DS/039", "https://linkedin.com/in/ganiyatadekunle/", "https://github.com/Abbiefied/"),
        ("Oluwatoyin Amodu",           "SCA/APC3/DS/048", "https://linkedin.com/in/oluwatoyin-amodu", "https://github.com/Abbiefied/disaster-risk-app"),
        ("Ifedigbo Ifeoma Christabel", "SCA/APC3/DS/195", "https://linkedin.com/in/ifedigbo-ifeoma-a52988264", "https://github.com/Abbiefied/disaster-risk-app"),
        ("Abigail Dahunsi",            "SCA/APC3/DS/078", "https://linkedin.com/in/abigail-dahunsi", "https://github.com/Abbiefied/disaster-risk-app"),
        ("Bunmi Apata",                "SCA/APC3/DS/081", "https://linkedin.com/in/bunmi-apata", "https://github.com/Abbiefied/disaster-risk-app"),
        ("Ogechi Obidile",             "SCA/APC3/DS/086", "https://linkedin.com/in/ogechi-obidile", "https://github.com/Abbiefied/disaster-risk-app"),
        ("Mistura Bakare",             "SCA/APC3/DS/139", "https://linkedin.com/in/mistura-bakare", "https://github.com/Abbiefied/disaster-risk-app"),
        ("Queen Abiche",               "SCA/APC3/DS/142", "https://linkedin.com/in/queen-abiche", "https://github.com/Abbiefied/disaster-risk-app"),
        ("Priscilla Akinwale",         "SCA/APC3/DS/186", "https://linkedin.com/in/priscilla-akinwale", "https://github.com/AkinwalePriscilla"),
        ("Ability James",              "SCA/APC3/DS/060", "https://linkedin.com/in/ability-james", "https://github.com/Abbiefied/disaster-risk-app"),
    ]
 
    # Render 3 cards per row
    cols_per_row = 3
    for row_start in range(0, len(team_members), cols_per_row):
        row_members = team_members[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, (name, member_id, linkedin_url, github_url) in zip(cols, row_members):

            parts = name.split()
            initials = (parts[0][0] + parts[-1][0]).upper()

            col.markdown(f"""
            <a href="{linkedin_url}" target="_blank" style="text-decoration:none">
            <div class="team-card">

            <div class="team-avatar">{initials}</div>
            <div class="team-name">{name}</div>
            <div class="team-id">{member_id}</div>
            <div style="display:flex;justify-content:center;gap:0.5rem">

            <!-- LinkedIn -->
            <a href="{linkedin_url}" target="_blank" class="team-link">
                {LI_SVG} LinkedIn
            </a>

            <!-- GitHub -->
            <a href="{github_url}" target="_blank" class="team-link">
                {GH_SVG} GitHub
            </a>

            </div>

            </div>
            </a>
        """, unsafe_allow_html=True)
 
    #Project banner at bottom of team tab 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1f35, #12192e);
        border: 1px solid rgba(91,127,245,0.15);
        border-radius: 12px;
        padding: 1.6rem 2rem;
        text-align: center;
    ">
        <div style="font-size:0.7rem;letter-spacing:1.5px;text-transform:uppercase;
                    color:#3d5a8a;margin-bottom:0.5rem">Capstone Project</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.25rem;
                    color:#c8d0e8;margin-bottom:0.4rem">
            Predicting the Occurrence Rate of Natural Disasters in Asia
        </div>
        <div style="font-size:0.82rem;color:#5c6a8a">
            SheCodeAfrica · Cohort 3 Data Science Bootcamp 2026 &nbsp;·&nbsp;
            Dataset: EM-DAT (CRED, UCLouvain) &nbsp;·&nbsp;
            Focus: Trend Analysis &amp; Logistic Regression
        </div>
    </div>
    """, unsafe_allow_html=True)
 