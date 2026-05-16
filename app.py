# ================================
# RANDOM FOREST REGRESSION APP
# HOUSE PRICE PREDICTION - STREAMLIT
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import plot_tree

# ================================
# PAGE CONFIG
# ================================

st.set_page_config(
    page_title="Random Forest · House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --amber:        #c8842a;
    --amber-light:  #e8a94d;
    --amber-pale:   #f5deb3;
    --brick:        #8b4513;
    --deep:         #4a2000;
    --accent:       #d4813a;
    --radius-card:  14px;
    --radius-sm:    8px;
    --shadow-card:  0 2px 16px rgba(0,0,0,0.08);
    --shadow-hover: 0 6px 24px rgba(0,0,0,0.13);
    --transition:   0.22s ease;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #2a0e00 0%, #7a3010 45%, #c8842a 100%);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🏠🏡🏘️🏗️🏠";
    position: absolute; right: 32px; top: 50%;
    transform: translateY(-50%);
    font-size: 2.8rem; letter-spacing: 6px; opacity: 0.2;
}
.hero h1 { color:#fff; font-size:2rem; font-weight:700; margin:0 0 6px 0; letter-spacing:-0.5px; }
.hero p  { color:#f5deb3; font-size:1.05rem; margin:0; font-weight:400; }

/* ── Section heading ── */
.section-heading {
    font-size:1.25rem; font-weight:700;
    margin:28px 0 14px 0; padding-left:12px;
    border-left:4px solid #c8842a;
    color:inherit; letter-spacing:-0.3px;
}

/* ── Cards ── */
.card {
    background: var(--card-bg, #fff);
    border: 1px solid var(--card-border, rgba(0,0,0,0.07));
    border-radius: var(--radius-card);
    padding: 22px 24px; margin-bottom:18px;
    box-shadow: var(--shadow-card);
    transition: box-shadow var(--transition);
}
.card:hover { box-shadow: var(--shadow-hover); }

/* ── Concept badge ── */
.concept-badge {
    display:inline-flex; align-items:flex-start; gap:10px;
    background: var(--badge-bg, #fdf6ec);
    border: 1px solid #e8c88a;
    border-radius: var(--radius-card);
    padding:14px 18px; margin-bottom:12px;
    width:100%; box-sizing:border-box;
}
.concept-badge .icon { font-size:1.4rem; flex-shrink:0; }
.concept-badge .text b { display:block; font-size:0.92rem; font-weight:700; margin-bottom:3px; color:#4a2000; }
.concept-badge .text span { font-size:0.85rem; color:var(--body-muted,#555); line-height:1.5; }

/* ── Advantage grid ── */
.adv-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:8px; }
.adv-item {
    display:flex; align-items:center; gap:9px;
    background: var(--chip-bg, #fdf0e0);
    border-radius:var(--radius-sm); padding:11px 14px;
    font-size:0.88rem; font-weight:500;
    color:var(--body-color,#222); border:1px solid #e8c88a;
}

/* ── Step pill ── */
.step-row { display:flex; align-items:flex-start; gap:14px; margin-bottom:14px; }
.step-num {
    background:#c8842a; color:#fff; border-radius:50%;
    min-width:32px; height:32px; display:flex;
    align-items:center; justify-content:center;
    font-size:0.85rem; font-weight:700; flex-shrink:0; margin-top:2px;
}
.step-body b { font-weight:700; font-size:0.93rem; display:block; margin-bottom:3px; color:var(--body-color,#222); }
.step-body span { font-size:0.84rem; color:var(--body-muted,#555); line-height:1.5; }

/* ── Analogy box ── */
.analogy-box {
    background: linear-gradient(135deg, #fde8c8, #f5deb3);
    border-radius:var(--radius-card); padding:18px 22px;
    margin:16px 0; border-left:4px solid #c8842a;
}
.analogy-box p { margin:0; color:#4a2000; font-size:0.92rem; line-height:1.6; }

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #fde8c8, #f5deb3);
    border:2px solid #c8842a; border-radius:var(--radius-card);
    padding:28px 32px; text-align:center;
}
.result-box h2 { color:#4a2000; margin:0 0 8px 0; font-size:1.8rem; }
.result-box p  { color:#6b3010; margin:4px 0; font-size:1rem; }

/* ── Price bar ── */
.prob-bar-wrap {
    background:rgba(0,0,0,0.08); border-radius:50px;
    height:10px; margin:12px 0 4px 0; overflow:hidden;
}
.prob-bar-fill {
    height:100%; border-radius:50px;
    background:linear-gradient(90deg,#c8842a,#4a2000);
    transition:width 0.5s ease;
}

/* ── Column info card ── */
.col-info-card {
    background:var(--card-bg,#fff);
    border:1px solid var(--card-border,rgba(0,0,0,0.07));
    border-radius:var(--radius-sm); padding:13px 15px; margin-bottom:10px;
}
.col-info-card .col-name { font-weight:700; font-size:0.88rem; color:#8b4513; margin-bottom:3px; }
.col-info-card .col-desc { font-size:0.82rem; color:var(--body-muted,#555); line-height:1.45; }

/* ── Expert card ── */
.expert-card {
    background:var(--badge-bg,#fdf6ec);
    border:1px solid #e8c88a; border-radius:var(--radius-card);
    padding:16px 18px; margin-bottom:10px; text-align:center;
}
.expert-card .expert-title { font-weight:700; color:#4a2000; font-size:0.9rem; margin-bottom:6px; }
.expert-card .expert-checks { font-size:0.82rem; color:var(--body-muted,#555); line-height:1.6; }
.expert-card .expert-price { font-size:1rem; font-weight:700; color:#c8842a; margin-top:8px; }

/* ── Dark mode overrides ── */
@media (prefers-color-scheme: dark) {
    :root {
        --card-bg:   #2a1a0a;
        --card-border: rgba(255,255,255,0.08);
        --chip-bg:   #3a2210;
        --badge-bg:  #3a2210;
        --body-color:#f5e6d3;
        --body-muted:#c8a070;
    }
    .concept-badge .text b { color:#e8a94d; }
    .analogy-box { background:linear-gradient(135deg,#3a1a00,#4a2a10); }
    .analogy-box p { color:#f5deb3; }
    .adv-item { background:#3a2210; border-color:#6a4020; color:#f5e6d3; }
    .step-body b { color:#f5e6d3; }
    .step-body span { color:#c8a070; }
    .section-heading { color:#e8a94d; }
    .col-info-card .col-desc { color:#c8a070; }
    .result-box { background:linear-gradient(135deg,#3a1a00,#4a2a10); border-color:#c8842a; }
    .result-box h2 { color:#f5deb3; }
    .result-box p  { color:#e8a94d; }
    .expert-card { background:#3a2210; border-color:#6a4020; }
    .expert-card .expert-title { color:#f5deb3; }
}

/* ── Widgets ── */
.stSlider label, .stSelectbox label, .stNumberInput label { font-weight:600; font-size:0.88rem; }

.stButton > button {
    background:linear-gradient(135deg,#8b4513,#c8842a);
    color:#fff; border:none; border-radius:var(--radius-sm);
    padding:10px 28px; font-size:0.95rem; font-weight:600;
    font-family:'DM Sans',sans-serif; cursor:pointer;
    transition:opacity var(--transition), transform var(--transition);
    box-shadow:0 3px 10px rgba(139,69,19,0.3);
}
.stButton > button:hover {
    opacity:0.88; transform:translateY(-1px);
    box-shadow:0 6px 18px rgba(139,69,19,0.35);
}

[data-testid="stMetric"] {
    background:var(--card-bg,#fff);
    border:1px solid var(--card-border,rgba(0,0,0,0.07));
    border-radius:var(--radius-card); padding:16px 20px;
    box-shadow:var(--shadow-card);
}

[data-testid="stSidebar"] { border-right:1px solid var(--card-border,rgba(0,0,0,0.06)); }
[data-testid="stDataFrame"] { border-radius:var(--radius-sm); overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# ================================
# HELPERS
# ================================

def fmt_inr(value):
    """Format number as Indian Rupees with Lakhs / Crore notation."""
    value = float(value)
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"

def apply_plot_theme(fig, ax_list=None):
    """Neutral theme — readable in both light & dark mode."""
    fig.patch.set_facecolor("none")
    if ax_list is None:
        axes_to_style = fig.axes
    else:
        axes_to_style = list(np.array(ax_list).flatten())
    for ax in axes_to_style:
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_color("#e8c88a")
        ax.tick_params(colors="#9e6030", labelsize=9)
        ax.xaxis.label.set_color("#9e6030")
        ax.yaxis.label.set_color("#9e6030")
        ax.title.set_color("#8b4513")

@st.cache_data
def load_default_data():
    """Load the default House Price dataset (embedded fallback)."""
    # Try uploaded file first
    for path in [
        "House_Price_Prediction_Dataset.csv",
        "/mnt/user-data/uploads/House_Price_Prediction_Dataset.csv",
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "Hoors" in df.columns:
                df.rename(columns={"Hoors": "Floors"}, inplace=True)
            return df

    # Embedded synthetic dataset (same schema, 200 rows)
    np.random.seed(42)
    n = 200
    locations  = ["Downtown", "Suburban", "Urban", "Rural"]
    conditions = ["Excellent", "Good", "Fair", "Poor"]
    garages    = ["Yes", "No"]

    area      = np.random.randint(500, 5000, n)
    bedrooms  = np.random.randint(1, 6, n)
    bathrooms = np.random.randint(1, 5, n)
    floors    = np.random.randint(1, 4, n)
    year_built = np.random.randint(1900, 2024, n)
    location  = np.random.choice(locations, n)
    condition = np.random.choice(conditions, n)
    garage    = np.random.choice(garages, n)

    loc_mult  = {"Downtown": 1.8, "Suburban": 1.3, "Urban": 1.5, "Rural": 1.0}
    cond_mult = {"Excellent": 1.3, "Good": 1.1, "Fair": 0.9, "Poor": 0.75}
    gar_mult  = {"Yes": 1.08, "No": 1.0}

    price = (
        area * 350
        + bedrooms * 80000
        + bathrooms * 60000
        + floors * 50000
        + (year_built - 1900) * 2000
        + np.array([loc_mult[l] for l in location]) * 50000
        + np.array([cond_mult[c] for c in condition]) * 40000
        + np.array([gar_mult[g] for g in garage]) * 30000
        + np.random.normal(0, 150000, n)
    ).clip(50000).astype(int)

    df = pd.DataFrame({
        "Id": range(1, n+1),
        "Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
        "Floors": floors, "YearBuilt": year_built,
        "Location": location, "Condition": condition,
        "Garage": garage, "Price": price,
    })
    return df

# ================================
# HERO
# ================================

st.markdown("""
<div class="hero">
    <h1>🏠 Random Forest Regression</h1>
    <p>House Price Prediction · Machine Learning Studio · Prices in Indian Rupees ₹</p>
</div>
""", unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================

with st.sidebar:
    st.markdown("### 📌 Navigation")
    section = st.radio(
        "Go to section",
        ("🌳 What is Random Forest Regression?", "🏡 House Price Prediction"),
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 📂 Dataset")
    uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
    st.markdown("---")
    st.markdown(
        "<small style='color:#c8842a;'>Built with Streamlit · Scikit-learn</small>",
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════
# SECTION 1 — WHAT IS RANDOM FOREST REGRESSION?
# ════════════════════════════════════════════════

if section == "🌳 What is Random Forest Regression?":

    st.markdown('<div class="section-heading">📖 What is Random Forest Regression?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p style="font-size:0.97rem;line-height:1.8;margin:0;">
            <strong>Random Forest Regression</strong> is a machine learning algorithm that builds
            <strong>many decision trees</strong> and <strong>averages their predictions</strong> to estimate
            a continuous number — like the price of a house.
            Instead of relying on a single model (which may be biased), it consults hundreds of trees
            and takes the average of all their estimates for a stable, accurate final prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="analogy-box">
        <p>
        🏠 <b>Real-life analogy:</b> Imagine you want to know how much your house is worth.
        Instead of asking one real estate agent (who might be biased or wrong), you ask
        <b>100 different property experts</b>. Each expert looks at different features of the house
        and gives their own price estimate. You then <b>average all their estimates</b> —
        that's exactly what Random Forest Regression does with decision trees.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Expert analogy visual
    st.markdown('<div class="section-heading">👨‍💼 How Multiple Experts Estimate Price</div>', unsafe_allow_html=True)

    experts = [
        ("Expert 1 🏗️", "Area\nBedrooms", "₹48.5 L"),
        ("Expert 2 🚿", "Bathrooms\nLocation", "₹52.0 L"),
        ("Expert 3 🔧", "Garage\nCondition", "₹49.8 L"),
        ("Expert 4 📅", "YearBuilt\nFloors", "₹51.2 L"),
    ]
    cols = st.columns(4)
    for col, (title, checks, price) in zip(cols, experts):
        with col:
            st.markdown(f"""
            <div class="expert-card">
                <div class="expert-title">{title}</div>
                <div class="expert-checks">Checks:<br>{checks}</div>
                <div class="expert-price">Predicts: {price}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="analogy-box" style="text-align:center;">
        <p><b>🌲 Forest averages all predictions →</b>
        Final Predicted Price = <b>₹50.38 Lakhs</b><br>
        <span style="font-size:0.85rem;">(48.5 + 52.0 + 49.8 + 51.2) ÷ 4 = 50.375</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Classification vs Regression
    st.markdown('<div class="section-heading">🏷️ Classification vs Regression</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="concept-badge">
            <div class="icon">🚢</div>
            <div class="text">
                <b>Classification (e.g. Titanic)</b>
                <span>Predicts a <em>category</em> — Survived or Not Survived.<br>
                Trees <b>vote</b> for a class. Majority wins.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="concept-badge">
            <div class="icon">💰</div>
            <div class="text">
                <b>Regression (e.g. House Price)</b>
                <span>Predicts a <em>number</em> — ₹52,00,000.<br>
                Trees output prices. Forest <b>averages</b> them.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="section-heading">⚙️ How It Works — Step by Step</div>', unsafe_allow_html=True)
    steps = [
        ("Collect Data", "Feed the model your dataset: Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage, and Price."),
        ("Build Many Trees", "The algorithm creates hundreds of decision trees. Each tree trains on a slightly different random sample of the data."),
        ("Each Tree Predicts a Price", "For a new house, every tree estimates its own price independently based on what it learned."),
        ("Average the Predictions", "All tree predictions are averaged. If 100 trees estimate ₹48L on average — that's the final predicted house price."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-row">
            <div class="step-num">{i}</div>
            <div class="step-body"><b>{title}</b><span>{desc}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Metrics explanation
    st.markdown('<div class="section-heading">📏 Regression Metrics Explained</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="concept-badge">
            <div class="icon">📉</div>
            <div class="text">
                <b>MAE — Mean Absolute Error</b>
                <span>Average gap between predicted and actual price. E.g. "On average, predictions are off by ₹2.5L." Lower = better.</span>
            </div>
        </div>
        <div class="concept-badge">
            <div class="icon">📐</div>
            <div class="text">
                <b>RMSE — Root Mean Squared Error</b>
                <span>Like MAE, but penalises large errors more heavily. Useful for spotting big mis-predictions.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="concept-badge">
            <div class="icon">🎯</div>
            <div class="text">
                <b>R² Score — Coefficient of Determination</b>
                <span>How much price variance the model explains. 1.0 = perfect. 0.85 = model explains 85% of variation.</span>
            </div>
        </div>
        <div class="concept-badge">
            <div class="icon">💹</div>
            <div class="text">
                <b>MAPE — Mean Absolute Percentage Error</b>
                <span>Average % error. E.g. "Predictions are within 12% of the actual price on average."</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Advantages
    st.markdown('<div class="section-heading">✅ Why Use Random Forest for House Prices?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="adv-grid">
        <div class="adv-item">🎯 <span><b>High Accuracy</b> — Handles complex pricing patterns</span></div>
        <div class="adv-item">🛡️ <span><b>Robust</b> — Not fooled by outlier property prices</span></div>
        <div class="adv-item">📦 <span><b>Mixed Data</b> — Works with both numbers &amp; categories</span></div>
        <div class="adv-item">🔍 <span><b>Feature Importance</b> — Reveals what drives price most</span></div>
        <div class="adv-item">🧩 <span><b>Missing Values</b> — Handles imperfect real-world data</span></div>
        <div class="adv-item">⚡ <span><b>No Assumptions</b> — No need for linear relationships</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Visual diagram
    st.markdown('<div class="section-heading">🎨 Visual: How the Forest Predicts Price</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    # Input
    inp = FancyBboxPatch((0.1, 1.7), 1.6, 1.6, boxstyle="round,pad=0.1",
                         facecolor="#fde8c8", edgecolor="#c8842a", linewidth=2)
    ax.add_patch(inp)
    ax.text(0.9, 2.5, "🏠\nHouse\nData", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#4a2000")

    # Trees
    tree_x = [2.8, 3.9, 5.0]
    tree_colors = ["#f5deb3", "#e8c88a", "#daa560"]
    tree_prices = ["₹48.5L", "₹52.0L", "₹49.8L"]
    for tx, tc, tv in zip(tree_x, tree_colors, tree_prices):
        box = FancyBboxPatch((tx, 1.2), 1.35, 2.6, boxstyle="round,pad=0.1",
                             facecolor=tc, edgecolor="#c8842a", linewidth=1.5)
        ax.add_patch(box)
        ax.text(tx+0.67, 3.45, "🌲 Tree", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#4a2000")
        ax.text(tx+0.67, 2.55, "₹", ha="center", va="center",
                fontsize=28, color="#8b4513", alpha=0.3)
        ax.text(tx+0.67, 1.58, tv, ha="center", va="center",
                fontsize=8, color="#4a2000", fontweight="bold")
        ax.annotate("", xy=(tx+0.05, 2.5), xytext=(1.7, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#c8842a", lw=1.5))

    # Average box
    vbox = FancyBboxPatch((6.8, 1.5), 1.55, 2.0, boxstyle="round,pad=0.1",
                          facecolor="#c8842a", edgecolor="#8b4513", linewidth=2)
    ax.add_patch(vbox)
    ax.text(7.575, 2.75, "📊 Avg", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#fff")
    ax.text(7.575, 2.2, "Sum ÷ 3", ha="center", va="center",
            fontsize=8, color="#fde8c8")
    for tx in tree_x:
        ax.annotate("", xy=(6.8, 2.5), xytext=(tx+1.4, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#c8842a", lw=1.5))

    # Final
    fbox = FancyBboxPatch((8.6, 1.7), 1.3, 1.6, boxstyle="round,pad=0.1",
                          facecolor="#4a2000", edgecolor="#c8842a", linewidth=2)
    ax.add_patch(fbox)
    ax.text(9.25, 2.7, "🏷️", ha="center", va="center", fontsize=16)
    ax.text(9.25, 2.2, "₹50.1L", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#f5deb3")
    ax.annotate("", xy=(8.6, 2.5), xytext=(8.35, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#4a2000", lw=2))
    ax.text(5, 0.45, "Multiple trees predict → average → final price estimate",
            ha="center", va="center", fontsize=9, color="#9e6030", style="italic")

    st.pyplot(fig, use_container_width=True)


# ════════════════════════════════════════════════
# SECTION 2 — HOUSE PRICE PREDICTION
# ════════════════════════════════════════════════

elif section == "🏡 House Price Prediction":

    # ── Load Dataset ─────────────────────────────
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        if "Hoors" in df_raw.columns:
            df_raw.rename(columns={"Hoors": "Floors"}, inplace=True)
        st.sidebar.success("Custom dataset loaded ✓")
        dataset_label = f"Uploaded: {uploaded_file.name}"
    else:
        df_raw = load_default_data()
        st.sidebar.success("Default House Price dataset loaded successfully ✓")
        dataset_label = "House Price Prediction Dataset"

    TARGET = "Price"

    # ── Dataset Overview ──────────────────────────
    st.markdown('<div class="section-heading">📊 Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",         df_raw.shape[0])
    c2.metric("Columns",      df_raw.shape[1])
    c3.metric("Median Price", fmt_inr(df_raw[TARGET].median()))
    c4.metric("Missing Cells", int(df_raw.isnull().sum().sum()))

    ca, cb = st.columns([3, 2])
    with ca:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Dataset Preview**")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Statistical Summary**")
        st.dataframe(df_raw.describe().round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Column Guide ──────────────────────────────
    st.markdown('<div class="section-heading">📘 Understanding Dataset Columns</div>', unsafe_allow_html=True)

    col_info = {
        "Id":        ("🔑", "Unique house identification number. Dropped before modeling — not a predictor."),
        "Area":      ("📐", "Total built-up area of the house in square feet. Larger area generally means higher price."),
        "Bedrooms":  ("🛏️",  "Number of bedrooms. More bedrooms usually increases property value."),
        "Bathrooms": ("🚿", "Number of bathrooms. A key comfort feature that adds to house value."),
        "Floors":    ("🏢", "Number of floors/storeys. Multi-floor homes tend to command higher prices."),
        "YearBuilt": ("📅", "Year the house was constructed. Newer builds may fetch higher prices."),
        "Location":  ("📍", "Area or locality — Downtown, Suburban, Urban, or Rural. One of the strongest price drivers."),
        "Condition": ("🔧", "Overall physical condition: Excellent, Good, Fair, or Poor."),
        "Garage":    ("🚗", "Whether a garage is available (Yes / No). Adds to convenience and property value."),
        "Price":     ("🎯", "TARGET VARIABLE — House price in Indian Rupees (₹). This is what the model predicts."),
    }

    cols3 = st.columns(3)
    for i, (col_name, (icon, desc)) in enumerate(col_info.items()):
        if col_name in df_raw.columns:
            with cols3[i % 3]:
                st.markdown(f"""
                <div class="col-info-card">
                    <div class="col-name">{icon} {col_name}</div>
                    <div class="col-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Missing Values ────────────────────────────
    st.markdown('<div class="section-heading">🧹 Missing Value Handling</div>', unsafe_allow_html=True)

    mv_before = df_raw.isnull().sum()
    df = df_raw.copy()

    # Fill missing
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    c_bef, c_aft = st.columns(2)
    with c_bef:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Before Handling**")
        st.dataframe(mv_before.rename("Missing"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c_aft:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**After Handling**")
        st.dataframe(df.isnull().sum().rename("Missing"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 View handling code"):
        st.code("""
# Numerical columns → fill with median
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns → fill with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
""", language="python")

    # ── Visualizations ────────────────────────────
    st.markdown('<div class="section-heading">📈 Data Visualizations</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Correlation Heatmap", "Countplots", "Pairplot"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df[TARGET] / 1e5, bins=35, color="#c8842a", edgecolor="#8b4513", alpha=0.85)
        axes[0].set_title("Price Distribution (₹ Lakhs)", fontsize=10, color="#8b4513")
        axes[0].set_xlabel("Price (₹ Lakhs)")
        axes[1].hist(np.log1p(df[TARGET]), bins=35, color="#e8a94d", edgecolor="#8b4513", alpha=0.85)
        axes[1].set_title("Log Price Distribution", fontsize=10, color="#8b4513")
        axes[1].set_xlabel("log(Price + 1)")
        apply_plot_theme(fig, axes)
        st.pyplot(fig, use_container_width=True)

    with tab2:
        num_df = df.select_dtypes(include=np.number).drop(columns=["Id"], errors="ignore")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f",
                    cmap=sns.diverging_palette(20, 40, s=80, as_cmap=True),
                    ax=ax, linewidths=0.5, linecolor="#fde8c8")
        ax.set_title("Correlation Heatmap", fontsize=11, color="#8b4513")
        apply_plot_theme(fig, [ax])
        st.pyplot(fig, use_container_width=True)

    with tab3:
        cat_plot_cols = [c for c in ["Location", "Condition", "Garage"] if c in df.columns]
        if cat_plot_cols:
            fig, axes = plt.subplots(1, len(cat_plot_cols), figsize=(5 * len(cat_plot_cols), 4))
            if len(cat_plot_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, cat_plot_cols):
                order = df[col].value_counts().index
                sns.countplot(x=df[col], ax=ax, order=order,
                              palette=sns.light_palette("#c8842a", n_colors=len(order)))
                ax.set_title(f"{col} Countplot", fontsize=10, color="#8b4513")
                ax.set_xlabel(col)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
            apply_plot_theme(fig, axes)
            st.pyplot(fig, use_container_width=True)

    with tab4:
        pair_cols = [c for c in ["Area", "Bedrooms", "Bathrooms", TARGET] if c in df.columns]
        fig = sns.pairplot(df[pair_cols], plot_kws={"color": "#c8842a", "alpha": 0.4},
                           diag_kws={"color": "#c8842a"})
        fig.fig.suptitle("Pairplot: Key Features vs Price", y=1.02, color="#8b4513")
        st.pyplot(fig.fig, use_container_width=True)

    # ── Outlier Detection ─────────────────────────
    st.markdown('<div class="section-heading">📦 Outlier Detection</div>', unsafe_allow_html=True)

    numeric_plot = [c for c in df.select_dtypes(include=np.number).columns
                    if c not in ["Id", TARGET]]

    tab_bef, tab_aft = st.tabs(["Before Removal", "After Removal"])

    with tab_bef:
        st.markdown(f"**Shape Before:** {df.shape[0]} rows × {df.shape[1]} columns")
        n_cols_box = min(len(numeric_plot), 4)
        n_rows_box = (len(numeric_plot) + n_cols_box - 1) // n_cols_box
        fig, axes = plt.subplots(n_rows_box, n_cols_box,
                                 figsize=(4 * n_cols_box, 3.5 * n_rows_box))
        axes = np.array(axes).flatten()
        for i, col in enumerate(numeric_plot):
            sns.boxplot(y=df[col], ax=axes[i], color="#e8c88a")
            axes[i].set_title(col, fontsize=9, color="#8b4513")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Outliers Before Handling", fontsize=11, color="#8b4513", y=1.02)
        plt.tight_layout()
        apply_plot_theme(fig, axes[:len(numeric_plot)])
        st.pyplot(fig, use_container_width=True)

    df_clean = df.copy()
    for col in numeric_plot:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[
            (df_clean[col] >= Q1 - 1.5 * IQR) &
            (df_clean[col] <= Q3 + 1.5 * IQR)
        ]

    with tab_aft:
        st.markdown(f"**Shape After:** {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
        fig, axes = plt.subplots(n_rows_box, n_cols_box,
                                 figsize=(4 * n_cols_box, 3.5 * n_rows_box))
        axes = np.array(axes).flatten()
        for i, col in enumerate(numeric_plot):
            sns.boxplot(y=df_clean[col], ax=axes[i], color="#c8842a")
            axes[i].set_title(col, fontsize=9, color="#8b4513")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Outliers After Handling", fontsize=11, color="#8b4513", y=1.02)
        plt.tight_layout()
        apply_plot_theme(fig, axes[:len(numeric_plot)])
        st.pyplot(fig, use_container_width=True)

    # ── Preprocessing ─────────────────────────────
    st.markdown('<div class="section-heading">⚙️ Preprocessing Pipeline</div>', unsafe_allow_html=True)

    df_model = df_clean.drop(columns=["Id"], errors="ignore")
    df_model = pd.get_dummies(df_model, columns=["Location", "Condition", "Garage"], drop_first=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Processed Dataset Preview** (after encoding)")
    st.dataframe(df_model.head(5), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 View preprocessing code"):
        st.code("""
# Drop ID column
df = df.drop("Id", axis=1)

# Encode categorical columns
df = pd.get_dummies(df, columns=["Location","Condition","Garage"], drop_first=True)

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]
""", language="python")

    # ── Features & Target ──────────────────────────
    X = df_model.drop(TARGET, axis=1)
    y = df_model[TARGET]

    # ── Feature Scaling ────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / Test Split ─────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    st.markdown('<div class="section-heading">✂️ Train / Test Split & Feature Scaling</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features",         X.shape[1])
    c2.metric("Training Samples", X_train.shape[0])
    c3.metric("Test Samples",     X_test.shape[0])
    c4.metric("Target",           "Price (₹)")

    with st.expander("📋 View split & scaling code"):
        st.code("""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
# Note: Random Forest does not strictly need scaling,
# but it is shown here for preprocessing completeness.
""", language="python")

    # ── Model Training ─────────────────────────────
    st.markdown('<div class="section-heading">🤖 Model Training</div>', unsafe_allow_html=True)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("✅ RandomForestRegressor trained successfully with 100 trees!")

    with st.expander("📋 View training code"):
        st.code("""
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
""", language="python")

    # ── Actual vs Predicted ────────────────────────
    st.markdown('<div class="section-heading">📋 Actual vs Predicted Prices</div>', unsafe_allow_html=True)

    comparison = pd.DataFrame({
        "Actual Price (₹)":    y_test.values[:10],
        "Predicted Price (₹)": y_pred[:10].astype(int),
        "Difference (₹)":      (y_test.values[:10] - y_pred[:10]).astype(int),
        "Error %":             (np.abs(y_test.values[:10] - y_pred[:10]) / y_test.values[:10] * 100).round(1),
    })
    comparison["Actual Price (₹)"]    = comparison["Actual Price (₹)"].apply(lambda x: fmt_inr(x))
    comparison["Predicted Price (₹)"] = comparison["Predicted Price (₹)"].apply(lambda x: fmt_inr(x))
    comparison["Difference (₹)"]      = comparison["Difference (₹)"].apply(lambda x: fmt_inr(abs(x)))

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**First 10 Predictions vs Actual** — closer values = better model performance")
    st.dataframe(comparison, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Evaluation ──────────────────────────
    st.markdown('<div class="section-heading">📊 Model Evaluation</div>', unsafe_allow_html=True)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test.clip(lower=1))) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R² Score",  f"{r2:.4f}",       help="Closer to 1.0 = better. Explains % of price variance.")
    c2.metric("MAE",       fmt_inr(mae),       help="Average absolute prediction error.")
    c3.metric("MSE",       fmt_inr(mse),       help="Mean squared error.")
    c4.metric("RMSE",      fmt_inr(rmse),      help="Root mean squared error — penalises large errors.")
    c5.metric("MAPE",      f"{mape:.1f}%",     help="Average % prediction error.")

    # Scatter + Residual
    col_sc, col_rs = st.columns(2)
    with col_sc:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test / 1e5, y_pred / 1e5, alpha=0.45, color="#c8842a",
                   edgecolors="#8b4513", linewidths=0.3, s=22)
        lims = [min(y_test.min(), y_pred.min()) / 1e5,
                max(y_test.max(), y_pred.max()) / 1e5]
        ax.plot(lims, lims, "r--", linewidth=1.5, alpha=0.7, label="Perfect Prediction")
        ax.set_xlabel("Actual Price (₹L)"); ax.set_ylabel("Predicted Price (₹L)")
        ax.set_title("Actual vs Predicted", fontsize=11, color="#8b4513")
        ax.legend(fontsize=8)
        apply_plot_theme(fig, [ax])
        st.pyplot(fig, use_container_width=True)

    with col_rs:
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_pred / 1e5, residuals / 1e5, alpha=0.45, color="#e8a94d",
                   edgecolors="#8b4513", linewidths=0.3, s=22)
        ax.axhline(0, color="#c8842a", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Predicted Price (₹L)"); ax.set_ylabel("Residual (₹L)")
        ax.set_title("Residual Plot", fontsize=11, color="#8b4513")
        apply_plot_theme(fig, [ax])
        st.pyplot(fig, use_container_width=True)

    # ── Feature Importance ────────────────────────
    st.markdown('<div class="section-heading">⭐ Feature Importance</div>', unsafe_allow_html=True)

    importance = (
        pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(importance) * 0.38)))
    colors = sns.light_palette("#c8842a", n_colors=len(importance))
    ax.barh(importance["Feature"], importance["Importance"],
            color=colors, edgecolor="none", height=0.65)
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_title("Feature Importance Ranking — What Drives House Price?",
                 fontsize=11, color="#8b4513", pad=10)
    for i, val in enumerate(importance["Importance"]):
        ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=7.5, color="#8b4513")
    apply_plot_theme(fig, [ax])
    st.pyplot(fig, use_container_width=True)

    # ── Interactive Tree Depth ─────────────────────
    st.markdown(
        '<div class="section-heading">🌳 Interactive Tree Depth Visualisation</div>',
        unsafe_allow_html=True
    )

    depth = st.slider("🎚️ Drag to change tree depth", min_value=1, max_value=8, value=3, step=1)

    depth_labels = {
        1: ("Very simple tree.", "⚡ Fast but may underfit"),
        2: ("Simple structure.", "Good for quick demos"),
        3: ("Balanced depth.", "✅ Recommended starting point"),
        4: ("Captures more patterns.", "Slightly more complex"),
        5: ("Complex tree.", "Watch for overfitting"),
        6: ("Very deep tree.", "⚠️ High variance risk"),
        7: ("Highly complex.", "⚠️ Overfitting likely"),
        8: ("Extreme depth.", "❌ Usually not ideal"),
    }
    lbl, note = depth_labels[depth]

    st.markdown(f"""
    <div style="background:rgba(200,132,42,0.08);padding:14px;border-radius:12px;
                border-left:5px solid #c8842a;margin-bottom:15px;">
    <b>Depth {depth}:</b> {lbl}<br>
    <span style="color:#e8a94d;">{note}</span>
    </div>
    """, unsafe_allow_html=True)

    model_depth = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=42)
    model_depth.fit(X_train, y_train)

    st.markdown("### 🌲 Actual Decision Tree Representation")

    fig, ax = plt.subplots(figsize=(22, 10))
    fig.patch.set_facecolor("#FBF3E8")
    ax.set_facecolor("#FBF3E8")

    plot_tree(
        model_depth.estimators_[0],
        feature_names=X.columns,
        filled=True, rounded=True,
        fontsize=8, max_depth=depth,
        impurity=True, proportion=True, precision=2,
        ax=ax
    )
    ax.set_title(
        f"Random Forest Regression Tree (Depth = {depth})",
        fontsize=18, color="#8b4513", pad=20, weight="bold"
    )
    st.pyplot(fig, use_container_width=True)

    # ════════════════════════════════════════════
    # HOUSE PRICE PREDICTOR
    # ════════════════════════════════════════════
    st.markdown('<div class="section-heading">🏠 House Price Predictor</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-bottom:20px;">
        <p style="margin:0;font-size:0.93rem;line-height:1.75;">
        Fill in the house details below and click <strong>Predict House Price</strong>.
        The model uses <strong>tree depth</strong> from the slider above —
        try changing depth to see how it affects the prediction estimate!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature descriptions shown above inputs
    st.markdown("""
    <div style="font-size:0.82rem;color:var(--body-muted,#777);margin-bottom:14px;line-height:1.8;">
    📐 <b>Area</b> — Total area in sq ft &nbsp;|&nbsp;
    🛏️ <b>Bedrooms</b> — No. of bedrooms &nbsp;|&nbsp;
    🚿 <b>Bathrooms</b> — No. of bathrooms &nbsp;|&nbsp;
    🏢 <b>Floors</b> — No. of floors &nbsp;|&nbsp;
    📅 <b>YearBuilt</b> — Construction year &nbsp;|&nbsp;
    📍 <b>Location</b> — Area type &nbsp;|&nbsp;
    🔧 <b>Condition</b> — Property condition &nbsp;|&nbsp;
    🚗 <b>Garage</b> — Garage availability
    </div>
    """, unsafe_allow_html=True)

    # Get actual ranges from dataset
    area_min = int(df_clean["Area"].min()) if "Area" in df_clean.columns else 300
    area_max = int(df_clean["Area"].max()) if "Area" in df_clean.columns else 5000
    area_med = int(df_clean["Area"].median()) if "Area" in df_clean.columns else 2500

    yr_min = int(df_clean["YearBuilt"].min()) if "YearBuilt" in df_clean.columns else 1900
    yr_max = int(df_clean["YearBuilt"].max()) if "YearBuilt" in df_clean.columns else 2024
    yr_med = int(df_clean["YearBuilt"].median()) if "YearBuilt" in df_clean.columns else 1970

    bed_opts = sorted(df_clean["Bedrooms"].unique().tolist()) if "Bedrooms" in df_clean.columns else [1,2,3,4,5]
    bath_opts = sorted(df_clean["Bathrooms"].unique().tolist()) if "Bathrooms" in df_clean.columns else [1,2,3,4]
    floor_opts = sorted(df_clean["Floors"].unique().tolist()) if "Floors" in df_clean.columns else [1,2,3]
    loc_opts = sorted(df_clean["Location"].unique().tolist()) if "Location" in df_clean.columns else ["Downtown","Suburban","Urban","Rural"]
    cond_opts = sorted(df_clean["Condition"].unique().tolist()) if "Condition" in df_clean.columns else ["Excellent","Good","Fair","Poor"]
    gar_opts = sorted(df_clean["Garage"].unique().tolist()) if "Garage" in df_clean.columns else ["Yes","No"]

    col1, col2, col3 = st.columns(3)

    with col1:
        area      = st.slider("📐 Area (sq ft)", area_min, area_max, area_med)
        bedrooms  = st.selectbox("🛏️ Bedrooms", bed_opts,
                                  index=min(2, len(bed_opts)-1))
        bathrooms = st.selectbox("🚿 Bathrooms", bath_opts,
                                  index=min(1, len(bath_opts)-1))

    with col2:
        floors    = st.selectbox("🏢 Floors", floor_opts)
        year_built = st.slider("📅 Year Built", yr_min, yr_max, yr_med)
        location  = st.selectbox("📍 Location", loc_opts)

    with col3:
        condition = st.selectbox("🔧 Condition", cond_opts)
        garage    = st.selectbox("🚗 Garage",    gar_opts,
                                  format_func=lambda x: f"{'✓ Yes' if x=='Yes' else '✗ No'} — Garage")

    # Build input row aligned to training columns
    input_row = pd.DataFrame({
        "Area":      [area],
        "Bedrooms":  [bedrooms],
        "Bathrooms": [bathrooms],
        "Floors":    [floors],
        "YearBuilt": [year_built],
        "Location":  [location],
        "Condition": [condition],
        "Garage":    [garage],
    })

    input_encoded = pd.get_dummies(input_row, columns=["Location", "Condition", "Garage"], drop_first=True)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    input_scaled  = scaler.transform(input_encoded)

    if st.button("🔮 Predict House Price"):
        prediction = model_depth.predict(input_scaled)[0]

        # Confidence interval from tree spread
        all_tree_preds = np.array([t.predict(input_scaled)[0] for t in model_depth.estimators_])
        pred_std  = all_tree_preds.std()
        pred_low  = max(0, prediction - 1.96 * pred_std)
        pred_high = prediction + 1.96 * pred_std

        st.markdown(f"""
        <div class="result-box">
            <h2>🏷️ Estimated House Price: {fmt_inr(prediction)}</h2>
            <p>95% Confidence Range: {fmt_inr(pred_low)} — {fmt_inr(pred_high)}</p>
            <p style="font-size:0.88rem;color:#9e6030;margin-top:8px;">
                Based on {len(model_depth.estimators_)} decision trees at depth {depth}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Price position bar relative to dataset range
        price_min = df_clean[TARGET].min()
        price_max = df_clean[TARGET].max()
        pct = min(100, max(0, (prediction - price_min) / (price_max - price_min) * 100))

        st.markdown(f"""
        <div style="margin-top:18px;">
            <div style="font-size:0.85rem;font-weight:600;margin-bottom:4px;color:#9e6030;">
                Price Position in Dataset Range &nbsp;
                (Min: {fmt_inr(price_min)} → Max: {fmt_inr(price_max)})
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{pct:.1f}%;"></div>
            </div>
            <div style="font-size:0.8rem;color:#9e6030;text-align:right;">{pct:.1f}% of price range</div>
        </div>
        """, unsafe_allow_html=True)

        # Tree voting summary metrics
        st.markdown("")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌲 Trees Used",        len(model_depth.estimators_))
        c2.metric("📊 Average Estimate",  fmt_inr(all_tree_preds.mean()))
        c3.metric("📉 Lowest Estimate",   fmt_inr(all_tree_preds.min()))
        c4.metric("📈 Highest Estimate",  fmt_inr(all_tree_preds.max()))

        # Distribution of tree estimates
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.hist(all_tree_preds / 1e5, bins=25, color="#c8842a",
                edgecolor="#8b4513", alpha=0.85)
        ax.axvline(prediction / 1e5, color="#4a2000", linewidth=2.5,
                   linestyle="--", label=f"Final Prediction: {fmt_inr(prediction)}")
        ax.set_xlabel("Price (₹ Lakhs)")
        ax.set_ylabel("Number of Trees")
        ax.set_title(
            f"Distribution of Tree Predictions ({len(model_depth.estimators_)} trees at depth {depth})",
            fontsize=10, color="#8b4513"
        )
        ax.legend(fontsize=8)
        apply_plot_theme(fig, [ax])
        st.pyplot(fig, use_container_width=True)

        st.markdown(f"""
        <div style="margin-top:12px;padding:14px 18px;background:rgba(200,132,42,0.08);
                    border-radius:10px;border-left:4px solid #c8842a;">
        <b>Interpretation:</b> The forest of <b>{len(model_depth.estimators_)} trees</b>
        estimated this house at an average of <b>{fmt_inr(all_tree_preds.mean())}</b>.
        The spread (std = {fmt_inr(pred_std)}) reflects how much the trees disagreed —
        a narrower spread means higher confidence.
        </div>
        """, unsafe_allow_html=True)