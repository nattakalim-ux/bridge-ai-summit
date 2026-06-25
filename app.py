"""
EpiGlass AI — Biological Age Platform
Streamlit demo with 3 pages: Calculator, Results, Simulation
"""

import streamlit as st
import numpy as np
import os, sys

# ── Path setup (must be before any local import) ─────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "Scripts")
MODEL_DIR   = os.path.join(BASE_DIR, "Model")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EpiGlass AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hide default Streamlit footer */
  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }

  /* Sidebar background */
  section[data-testid="stSidebar"] {
    background-color: #EEEDFE;
  }
  section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #3B35A8;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #FAFAFA;
    border: 1px solid #E0E0E0;
    border-radius: 12px;
    padding: 16px 20px;
  }

  /* Big result card */
  .result-card {
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;
    text-align: center;
  }
  .card-green  { background: #EAFAF1; border: 1.5px solid #1D9E75; }
  .card-yellow { background: #FFFDE7; border: 1.5px solid #F9A825; }
  .card-orange { background: #FFF3E0; border: 1.5px solid #EF6C00; }
  .card-red    { background: #FFF0F0; border: 1.5px solid #E24B4A; }

  div[data-testid="stExpander"] {
    border: 1px solid #D0CCFF;
    border-radius: 10px;
  }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LEVINE PHENOAGE FORMULA
# ═══════════════════════════════════════════════════════════════════════════════
def compute_phenoage(albumin, creatinine, glucose_mgdl, crp_mgl,
                     lymph_pct, mcv, rdw, alp, wbc, age):
    glucose_mmol = glucose_mgdl / 18.016
    crp_mgdl     = crp_mgl / 10.0
    crp_safe     = max(crp_mgdl, 0.001)
    xb = (-19.907
          - 0.0336  * albumin
          + 0.0095  * creatinine
          + 0.1953  * glucose_mmol
          + 0.0954  * np.log(crp_safe)
          - 0.0120  * lymph_pct
          + 0.0268  * mcv
          + 0.3306  * rdw
          + 0.00188 * alp
          + 0.0554  * wbc
          + 0.0804  * age)
    raw_M = 1 - np.exp(-1.51714 * np.exp(xb) / 0.0076927)
    M = np.clip(raw_M, 0.0001, 0.9999)
    phenoage = 141.50 + np.log(-0.00553 * np.log(1 - M)) / 0.09165
    gap = phenoage - age
    return round(float(phenoage), 1), round(float(gap), 1)

# ── Feature contributions (for top-3 biomarker breakdown) ────────────────────
LEVINE_COEFFS = {
    "Albumin":      -0.0336,
    "Creatinine":   +0.0095,
    "Glucose":      +0.1953,   # applied to mmol/L
    "CRP":          +0.0954,   # applied to ln(mg/dL)
    "Lymphocyte %": -0.0120,
    "MCV":          +0.0268,
    "RDW":          +0.3306,
    "ALP":          +0.00188,
    "WBC":          +0.0554,
    "Age":          +0.0804,
}

def feature_contributions(albumin, creatinine, glucose_mgdl, crp_mgl,
                           lymph_pct, mcv, rdw, alp, wbc, age):
    glucose_mmol = glucose_mgdl / 18.016
    crp_mgdl     = max(crp_mgl / 10.0, 0.001)
    vals = {
        "Albumin":      albumin,
        "Creatinine":   creatinine,
        "Glucose":      glucose_mmol,
        "CRP":          np.log(crp_mgdl),
        "Lymphocyte %": lymph_pct,
        "MCV":          mcv,
        "RDW":          rdw,
        "ALP":          alp,
        "WBC":          wbc,
        "Age":          age,
    }
    contribs = {k: LEVINE_COEFFS[k] * vals[k] for k in LEVINE_COEFFS}
    max_abs = max(abs(v) for v in contribs.values()) or 1
    return {k: v / max_abs for k, v in contribs.items()}

# ── Mortality model loader (cached) ──────────────────────────────────────────
@st.cache_resource
def load_mortality_model():
    try:
        import joblib
        base  = joblib.load(os.path.join(MODEL_DIR, "mortality_base_model.pkl"))
        ir    = joblib.load(os.path.join(MODEL_DIR, "mortality_ir.pkl"))
        return base, ir
    except Exception as e:
        return None, str(e)

def predict_mortality(albumin, creatinine, glucose_mgdl, crp_mgl,
                      lymph_pct, mcv, rdw, alp, wbc):
    base, ir = load_mortality_model()
    if base is None:
        return None, ir   # ir contains the error string

    # Mortality model uses log1p on: crp_mgl(raw mg/L), glucose(mmol/L), wbc, alp
    glucose_mmol = glucose_mgdl / 18.016
    import numpy as np
    X = np.array([[albumin, creatinine,
                   np.log1p(glucose_mmol),  # log1p(mmol/L) matches training
                   np.log1p(alp),
                   np.log1p(crp_mgl),       # log1p(mg/L) matches training
                   mcv, rdw,
                   np.log1p(wbc),
                   lymph_pct]])
    raw  = base.predict_proba(X)[0, 1]
    prob = float(ir.predict([raw])[0])
    return round(prob * 100, 1), None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 EpiGlass AI")
    st.markdown("*Biological Age Platform*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🔬 Calculator", "📊 Results", "🔮 Simulation"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Model: Levine PhenoAge (2018)")
    st.caption("Data: NHANES 2015-2016")
    st.caption("Mortality AUC: 0.8827")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔬 Calculator":
    st.title("EpiGlass AI — Know Your Biological Age")
    st.markdown("##### Enter your blood test results to calculate your true biological age")
    st.divider()

    # Example values button
    if st.button("⚡ Load Example Values (Healthy 35-year-old)", use_container_width=False):
        st.session_state.ex_albumin   = 4.5
        st.session_state.ex_creatinine = 0.8
        st.session_state.ex_glucose   = 88.0
        st.session_state.ex_crp       = 0.8
        st.session_state.ex_lymph     = 32.0
        st.session_state.ex_mcv       = 88.0
        st.session_state.ex_rdw       = 13.0
        st.session_state.ex_alp       = 65.0
        st.session_state.ex_wbc       = 6.0
        st.session_state.ex_age       = 35

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Blood Chemistry**")
        albumin    = st.slider("Albumin (g/dL)",         2.0,  6.0,  float(st.session_state.get("ex_albumin",    4.2)), 0.1)
        creatinine = st.slider("Creatinine (mg/dL)",     0.4,  3.0,  float(st.session_state.get("ex_creatinine", 0.9)), 0.05)
        glucose    = st.slider("Glucose (mg/dL)",        60.0, 300.0, float(st.session_state.get("ex_glucose",   95.0)), 1.0)
        crp        = st.slider("CRP (mg/L)",             0.1,  20.0, float(st.session_state.get("ex_crp",        1.5)), 0.1)
        lymph      = st.slider("Lymphocyte (%)",         5.0,  60.0, float(st.session_state.get("ex_lymph",     28.0)), 0.5)

    with col2:
        st.markdown("**Blood Count & Other**")
        mcv   = st.slider("MCV (fL)",                  60.0, 120.0, float(st.session_state.get("ex_mcv",  89.0)), 0.5)
        rdw   = st.slider("RDW (%)",                   10.0,  25.0, float(st.session_state.get("ex_rdw",  13.5)), 0.1)
        alp   = st.slider("ALP (U/L)",                 20.0, 300.0, float(st.session_state.get("ex_alp",  75.0)), 1.0)
        wbc   = st.slider("WBC (1000 cells/µL)",        1.0,  15.0, float(st.session_state.get("ex_wbc",   6.5)), 0.1)
        age   = st.slider("Age (years)",               18,    90,    int(st.session_state.get("ex_age",    45)))

    st.divider()
    calc_btn = st.button("🔬 Calculate My Biological Age →",
                         use_container_width=True, type="primary")

    if calc_btn:
        phenoage, gap = compute_phenoage(
            albumin, creatinine, glucose, crp, lymph, mcv, rdw, alp, wbc, age)
        mort_pct, mort_err = predict_mortality(
            albumin, creatinine, glucose, crp, lymph, mcv, rdw, alp, wbc)
        contribs = feature_contributions(
            albumin, creatinine, glucose, crp, lymph, mcv, rdw, alp, wbc, age)

        # Store in session state
        st.session_state.result = dict(
            age=age, phenoage=phenoage, gap=gap,
            mort_pct=mort_pct, mort_err=mort_err,
            contribs=contribs,
            inputs=dict(albumin=albumin, creatinine=creatinine, glucose=glucose,
                        crp=crp, lymph=lymph, mcv=mcv, rdw=rdw, alp=alp, wbc=wbc),
        )
        st.success("✅ Calculation complete! Navigate to **📊 Results** in the sidebar.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Results":
    st.title("Your Biological Age Results")

    if "result" not in st.session_state:
        st.info("👈 Go to **🔬 Calculator** first and press **Calculate**.")
        st.stop()

    r        = st.session_state.result
    age      = r["age"]
    phenoage = r["phenoage"]
    gap      = r["gap"]

    # Validate
    if phenoage < 0 or phenoage > 150:
        st.warning("⚠️ Unusual PhenoAge result — please check your input values.")

    # ── 3 metric cards ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Your Chronological Age", f"{age} yrs")
    c2.metric("Your Biological Age",    f"{phenoage} yrs")
    c3.metric("Aging Gap",              f"{gap:+.1f} yrs",
              delta=f"{gap:+.1f} yrs vs chronological age",
              delta_color="inverse")

    # ── Colored status card ───────────────────────────────────────────────────
    if gap < -2:
        card_cls = "card-green"
        status   = "🎉 You're biologically younger than your age!"
        detail   = f"Your body is {abs(gap):.1f} years younger than your calendar age."
    elif gap <= 3:
        card_cls = "card-yellow"
        status   = "✅ Normal aging range"
        detail   = "Your biological and chronological ages are well aligned."
    elif gap <= 7:
        card_cls = "card-orange"
        status   = "⚠️ Slightly accelerated aging"
        detail   = f"Your body is aging {gap:.1f} years faster than average."
    else:
        card_cls = "card-red"
        status   = "🔴 Significantly accelerated aging"
        detail   = f"Your body is aging {gap:.1f} years faster than your calendar age."

    st.markdown(f"""
    <div class="result-card {card_cls}">
      <h2 style="margin:0">{status}</h2>
      <p style="margin:6px 0 0">{detail}</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Mortality risk ─────────────────────────────────────────────────────────
    st.markdown("### 10-Year Mortality Risk")
    if r["mort_err"]:
        st.error(f"Mortality model loading error: {r['mort_err']}")
    elif r["mort_pct"] is not None:
        mp = r["mort_pct"]
        color = "#1D9E75" if mp < 5 else ("#F9A825" if mp < 15 else "#E24B4A")
        st.markdown(f"""
        <div style="font-size:1.6rem; font-weight:700; color:{color}; margin:8px 0">
          {mp}%
        </div>
        <p style="color:#666; margin:0">
          Your estimated 10-year all-cause mortality risk<br>
          <small>Based on NHANES 2015-2016 validation cohort &nbsp;|&nbsp;
          Elastic Net AUC-ROC = 0.8827</small>
        </p>
        """, unsafe_allow_html=True)
        if mp < 5:
            st.success("Low risk — keep up the healthy lifestyle!")
        elif mp < 15:
            st.warning("Moderate risk — consider lifestyle improvements.")
        else:
            st.error("Elevated risk — please consult a healthcare professional.")

    st.divider()

    # ── Top 3 contributing biomarkers ─────────────────────────────────────────
    st.markdown("### Top Biomarkers Driving Your Result")
    contribs = r["contribs"]
    sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    for feat, norm_val in sorted_c:
        direction = "↑ Accelerating aging" if norm_val > 0 else "↓ Protective"
        bar_color = "#E24B4A" if norm_val > 0 else "#1D9E75"
        bar_pct   = int(abs(norm_val) * 100)
        st.markdown(f"**{feat}** — {direction}")
        st.markdown(f"""
        <div style="background:#F0F0F0; border-radius:6px; height:12px; width:100%; margin-bottom:12px">
          <div style="background:{bar_color}; width:{bar_pct}%; height:12px; border-radius:6px"></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Research validation figures ────────────────────────────────────────────
    with st.expander("📊 View Research Validation", expanded=False):
        figs = [
            ("roc_curves_final.png",                "Model Performance — ROC Curves (Levine vs ML models)"),
            ("feature_importance_final.png",         "Biomarker Importance Comparison"),
            ("phenoage_scatter_final.png",            "Biological Age vs Chronological Age (NHANES cohort)"),
            ("mortality_distribution_final.png",      "Mortality Risk Distribution by Outcome"),
            ("phenoage_gap_distribution_final.png",   "Aging Gap Distribution — Died vs Survived"),
        ]
        for fname, caption in figs:
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                st.image(fpath, caption=caption, use_container_width=True)
            else:
                st.warning(f"Image not found: {fpath}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Simulation":
    st.title("Simulate Lifestyle Changes")
    st.markdown("##### See how improving your biomarkers could reduce your biological age")

    if "result" not in st.session_state:
        st.info("👈 Go to **🔬 Calculator** first and press **Calculate**.")
        st.stop()

    r   = st.session_state.result
    inp = r["inputs"]

    baseline_phenoage = r["phenoage"]
    baseline_gap      = r["gap"]
    age               = r["age"]

    st.markdown(f"**Baseline Biological Age:** {baseline_phenoage} yrs "
                f"(Gap: {baseline_gap:+.1f} yrs)")
    st.divider()

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("### Adjust Lifestyle Factors")
        gluc_red = st.slider("Reduce blood glucose (%)",         0, 30, 0, 1,
                             help="Effect of diet/exercise on fasting glucose")
        crp_red  = st.slider("Reduce inflammation — CRP (%)",   0, 50, 0, 1,
                             help="Effect of anti-inflammatory diet/exercise")
        alb_imp  = st.slider("Improve albumin (%)",             0, 15, 0, 1,
                             help="Effect of improved protein intake/nutrition")

    # Recalculate with modified values
    new_glucose = inp["glucose"] * (1 - gluc_red / 100)
    new_crp     = inp["crp"]    * (1 - crp_red  / 100)
    new_albumin = inp["albumin"] * (1 + alb_imp  / 100)
    new_crp     = max(new_crp, 0.01)

    proj_phenoage, proj_gap = compute_phenoage(
        new_albumin, inp["creatinine"], new_glucose, new_crp,
        inp["lymph"],  inp["mcv"], inp["rdw"], inp["alp"], inp["wbc"], age)

    age_reduction = round(baseline_phenoage - proj_phenoage, 1)

    with col_b:
        st.markdown("### Projected Result")
        st.metric("Original Biological Age",   f"{baseline_phenoage} yrs")
        st.metric("Projected Biological Age",  f"{proj_phenoage} yrs",
                  delta=f"{-age_reduction:+.1f} yrs",
                  delta_color="inverse")

        if age_reduction > 0:
            st.success(f"🎯 Potential age reduction: **{age_reduction} years younger!**")
        elif age_reduction == 0:
            st.info("Adjust the sliders above to see the projected impact.")
        else:
            st.warning("These changes would slightly increase biological age in this model.")

        # Visual gauge
        if age_reduction > 0:
            pct = min(age_reduction / 15 * 100, 100)
            st.markdown(f"""
            <div style="margin-top:12px">
              <p style="margin:0; font-size:0.85rem; color:#666">
                Potential age reduction progress (max ~15 yrs)
              </p>
              <div style="background:#E8E8E8; border-radius:8px; height:18px; margin-top:4px">
                <div style="background:#1D9E75; width:{pct:.0f}%; height:18px;
                            border-radius:8px; transition:width 0.3s"></div>
              </div>
              <p style="margin:4px 0 0; font-size:0.85rem; color:#1D9E75; font-weight:600">
                {age_reduction} yrs / ~15 yrs max estimated
              </p>
            </div>
            """, unsafe_allow_html=True)

        # Show what changed
        if gluc_red > 0 or crp_red > 0 or alb_imp > 0:
            st.markdown("**Changes applied:**")
            if gluc_red > 0:
                st.caption(f"  Glucose: {inp['glucose']:.0f} → {new_glucose:.0f} mg/dL (−{gluc_red}%)")
            if crp_red > 0:
                st.caption(f"  CRP: {inp['crp']:.1f} → {new_crp:.2f} mg/L (−{crp_red}%)")
            if alb_imp > 0:
                st.caption(f"  Albumin: {inp['albumin']:.1f} → {new_albumin:.2f} g/dL (+{alb_imp}%)")

    st.divider()
    st.caption(
        "⚠️ This simulation is for educational purposes only. "
        "Actual biological age changes depend on many factors. "
        "Consult a healthcare professional for medical advice."
    )
