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
    font-size: 2.4rem;
    font-weight: 400;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
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

/*Streamlit component overrides*/
div[data-testid="stSelectbox"] label,
div[data-testid="stMetricLabel"]   { color: #8892b0 !important; }
div[data-testid="stMetricValue"]   { color: #ffffff !important; font-size: 1.8rem !important; }
div[data-testid="stTabs"] button   {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #8892b0;
}
div[data-testid="stTabs"] button[aria-selected="true"] { color: #c8d0e8; }

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


#Hero banner
st.markdown("""
<div class="hero">
    <div class="hero-tag">🌏 Group 5 · SheCodeAfrica Bootcamp 2026</div>
    <p class="hero-title">Disaster Risk Intelligence System</p>
    <p class="hero-sub">
        Predicting the occurrence rate of natural disasters across Asia &nbsp;·&nbsp;
        Logistic Regression on EM-DAT panel data (2001–2024)
    </p>
</div>
""", unsafe_allow_html=True)


#Load data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

cy = load_data()


#Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Risk",
    "🏆 Country Rankings",
    "📈 Trend Analysis",
    "🧠 Model Insights",
])


# -----------------------------------------------------------------------------
# TAB 1 - PREDICTION
# -----------------------------------------------------------------------------
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

    #Current year ──
    with col_curr:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Current Year</p>', unsafe_allow_html=True)

        input_df = selected_row.drop(
            columns=["Country", "High_Occurrence", "event_count",
                     "total_deaths", "total_affected", "total_damage"],
            errors="ignore"
        )
        pred_curr, prob_curr = predict_risk(input_df)
        label_curr, colour_curr = risk_label(prob_curr)

        st.markdown(
            f'<p class="prob-number" style="color:{colour_curr}">'
            f'{prob_curr:.1%}</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="risk-badge" style="'
            f'background:{"rgba(231,76,60,0.15)" if pred_curr else "rgba(46,204,113,0.12)"}; '
            f'border:1px solid {colour_curr}; color:{colour_curr}">'
            f'{"⚠ " + label_curr + " Risk" if pred_curr else "✔ " + label_curr + " Risk"}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"**{country} - {year}**", )
        st.progress(prob_curr)
        st.markdown('</div>', unsafe_allow_html=True)

    #Next year
    with col_next:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">Next Year Forecast</p>',
                    unsafe_allow_html=True)

        next_row    = build_next_year_input(selected_row)
        next_input  = next_row.drop(
            columns=["Country", "High_Occurrence", "event_count",
                     "total_deaths", "total_affected", "total_damage"],
            errors="ignore"
        )
        pred_next, prob_next = predict_risk(next_input)
        label_next, colour_next = risk_label(prob_next)

        st.markdown(
            f'<p class="prob-number" style="color:{colour_next}">'
            f'{prob_next:.1%}</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="risk-badge" style="'
            f'background:{"rgba(231,76,60,0.15)" if pred_next else "rgba(46,204,113,0.12)"}; '
            f'border:1px solid {colour_next}; color:{colour_next}">'
            f'{"⚠ " + label_next + " Risk" if pred_next else "✔ " + label_next + " Risk"}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"**{country} - {year + 1} (projected)**")
        st.progress(prob_next)
        st.markdown('</div>', unsafe_allow_html=True)

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
        arrow   = "↑" if is_up else "↓"
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


# -----------------------------------------------------------------------------
# TAB 2 - COUNTRY RANKINGS
# -----------------------------------------------------------------------------
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
                use_container_width=True,    # FIX: original used width='stretch' (invalid)
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
                title=f"Top 15 Countries by Risk Probability ({selected_year})",
                xaxis_title="Risk Probability",
                plot_bgcolor="#161b2e",
                paper_bgcolor="#161b2e",
                font_color="#c8d0e8",
                height=520,
                margin=dict(l=10, r=60, t=50, b=10),
                xaxis=dict(range=[0, 1.1], gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig, use_container_width=True)  # FIX: invalid kwarg removed


# -----------------------------------------------------------------------------
# TAB 3 - TRENDS
# -----------------------------------------------------------------------------
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
        plot_bgcolor="#161b2e",
        paper_bgcolor="#161b2e",
        font_color="#c8d0e8",
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

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
            plot_bgcolor="#161b2e",
            paper_bgcolor="#161b2e",
            font_color="#c8d0e8",
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

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
        plot_bgcolor="#161b2e",
        paper_bgcolor="#161b2e",
        font_color="#c8d0e8",
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar=dict(tickfont=dict(color="#c8d0e8")),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4 - MODEL INSIGHTS
# -----------------------------------------------------------------------------
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
        title="Top 20 Feature Coefficients (by absolute magnitude)",
        plot_bgcolor="#161b2e",
        paper_bgcolor="#161b2e",
        font_color="#c8d0e8",
        height=560,
        margin=dict(l=10, r=80, t=50, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", autorange="reversed"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    #Methodology notes
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