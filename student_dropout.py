import datetime as dt

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# BRAND
# ============================================================
APP_NAME = "EduPulse"
TAGLINE = "Early-warning insights for student success"

LOGO_SVG = """
<svg width="{s}" height="{s}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0F8B8D"/><stop offset="1" stop-color="#12263A"/>
    </linearGradient>
  </defs>
  <rect width="64" height="64" rx="16" fill="url(#lg)"/>
  <path d="M32 12 L54 23 L32 34 L10 23 Z" fill="#F5F7FA"/>
  <path d="M20 29 V38 C20 42 26 45 32 45 C38 45 44 42 44 38 V29 L32 35 Z" fill="#F5F7FA" opacity=".92"/>
  <path d="M8 52 H22 L27 44 L33 58 L38 49 H56" fill="none" stroke="#F2A541"
        stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def logo(size=56):
    return LOGO_SVG.format(s=size)


st.set_page_config(
    page_title=f"{APP_NAME} | Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');
:root{--ink:#12263A;--teal:#0F8B8D;--amber:#F2A541;--paper:#F5F7FA;--coral:#D1495B;--green:#2E9E6B;--line:#E2E8F0;}
html, body, [class*="css"]{font-family:'Inter',sans-serif;}
h1,h2,h3,.brand-name,.hero-title{font-family:'Sora',sans-serif !important;}
.block-container{padding-top:1.6rem;max-width:1200px;}

.hero{display:flex;align-items:center;gap:22px;padding:26px 30px;border-radius:22px;
  background:linear-gradient(120deg,#12263A 0%,#17455A 60%,#0F8B8D 120%);color:#fff;margin-bottom:22px;}
.hero-title{font-size:36px;font-weight:800;margin:0;line-height:1.1;color:#fff;}
.hero-sub{font-size:16px;color:#CFE3E8;margin-top:6px;max-width:640px;}
.hero-badges span{display:inline-block;background:rgba(255,255,255,.12);border-radius:999px;
  padding:4px 12px;font-size:13px;margin:10px 8px 0 0;color:#fff;}

.card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin-bottom:14px;}
.kpi{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:14px 16px;}
.kpi .v{font-family:'Sora',sans-serif;font-size:26px;font-weight:700;color:var(--ink);}
.kpi .l{font-size:13px;color:#5B6B7B;}

.result{border-radius:18px;padding:22px 26px;color:#fff;margin:6px 0 16px 0;}
.result.Dropout{background:linear-gradient(120deg,#D1495B,#9E2F42);}
.result.Enrolled{background:linear-gradient(120deg,#E9A23B,#C77B14);}
.result.Graduate{background:linear-gradient(120deg,#2E9E6B,#0F6E52);}
.result .big{font-family:'Sora',sans-serif;font-size:34px;font-weight:800;margin:0;}
.result .sm{opacity:.92;font-size:15px;}

.bar-row{display:flex;align-items:center;gap:12px;margin:10px 0;}
.bar-row .name{width:90px;font-weight:600;color:var(--ink);}
.bar-bg{flex:1;height:14px;background:#E9EEF3;border-radius:999px;overflow:hidden;}
.bar-fill{height:100%;border-radius:999px;}
.bar-row .pct{width:60px;text-align:right;font-weight:600;color:var(--ink);}

.chip{display:inline-block;border-radius:999px;padding:5px 12px;font-size:13px;margin:4px 6px 4px 0;font-weight:500;}
.chip.good{background:#E3F4EC;color:#1B6E4B;}
.chip.warn{background:#FCEBD0;color:#8A5A0B;}
.chip.bad{background:#F9DDE1;color:#9E2F42;}

div.stButton > button{border-radius:12px;height:50px;font-weight:700;font-size:17px;}
[data-testid="stSidebar"]{background:#F0F4F8;}
.brand-name{font-size:22px;font-weight:800;color:var(--ink);margin:0;}
.brand-tag{font-size:12.5px;color:#5B6B7B;margin:0;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("student_dropout_model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Place `student_dropout_model.pkl` in the same "
        "folder as this app and reload."
    )
    st.stop()

LABELS = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
COLORS = {"Dropout": "#D1495B", "Enrolled": "#E9A23B", "Graduate": "#2E9E6B"}

# ============================================================
# SESSION STATE + PRESETS
# ============================================================
DEFAULTS = dict(
    age=20, admission_grade=120.0, course=171, tuition=1,
    first_grade=10.0, first_evaluations=5, first_approved=5,
    second_grade=10.0, second_evaluations=5, second_approved=5,
)

PRESETS = {
    "Strong performer": dict(age=18, admission_grade=155.0, course=171, tuition=1,
                             first_grade=15.0, first_evaluations=6, first_approved=6,
                             second_grade=15.5, second_evaluations=6, second_approved=6),
    "Average student": dict(DEFAULTS),
    "Struggling student": dict(age=24, admission_grade=100.0, course=171, tuition=0,
                               first_grade=7.0, first_evaluations=6, first_approved=2,
                               second_grade=5.0, second_evaluations=6, second_approved=1),
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)


def apply_preset():
    name = st.session_state.get("preset")
    if name in PRESETS:
        for k, v in PRESETS[name].items():
            st.session_state[k] = v


def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;">{logo(48)}'
        f'<div><p class="brand-name">{APP_NAME}</p>'
        f'<p class="brand-tag">{TAGLINE}</p></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("Quick start")
    st.selectbox(
        "Load a sample profile",
        options=list(PRESETS.keys()),
        index=None,
        placeholder="Choose a profile...",
        key="preset",
        on_change=apply_preset,
        help="Fills the form with example values so you can see how the model responds.",
    )
    st.button("Reset form", on_click=reset_form, use_container_width=True)

    st.divider()
    st.subheader("Session summary")
    n = len(st.session_state.history)
    st.metric("Predictions made", n)
    if n:
        counts = pd.Series([h["Prediction"] for h in st.session_state.history]).value_counts()
        for status, c in counts.items():
            st.caption(f"{status}: {c}")
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.last = None
            st.rerun()

    st.divider()
    st.caption("Machine-learning demo. Predictions support, but never replace, human judgement.")

# ============================================================
# HERO
# ============================================================
st.markdown(
    f"""
<div class="hero">
  <div>{logo(84)}</div>
  <div>
    <p class="hero-title">{APP_NAME}</p>
    <div class="hero-sub">Estimate whether a student is likely to <b>drop out</b>, stay
    <b>enrolled</b>, or <b>graduate</b> from admission details and first-year academic results.</div>
    <div class="hero-badges"><span>10 input features</span><span>3 outcome classes</span>
    <span>Instant probabilities</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_predict, tab_insights, tab_history, tab_guide = st.tabs(
    ["Predict", "Insights", "History", "Guide"]
)

# ============================================================
# TAB 1 - PREDICT
# ============================================================
with tab_predict:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Admission details")
        age = st.number_input("Age at enrollment", 15, 100, key="age", step=1,
                              help="Age when the student enrolled (15-100).")
        admission_grade = st.slider("Admission grade", 0.0, 200.0, key="admission_grade", step=0.5,
                                    help="Entry grade on a 0-200 scale.")
        course = st.number_input("Course code", 0, 10000, key="course", step=1,
                                 help="Numeric course code used in the training dataset. This is a code, not a grade.")
        tuition = st.radio(
            "Tuition fees up to date?", [1, 0], key="tuition", horizontal=True,
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    with col2:
        st.subheader("Academic performance")
        s1, s2 = st.tabs(["1st semester", "2nd semester"])
        with s1:
            first_grade = st.slider("Average grade", 0.0, 20.0, key="first_grade", step=0.1)
            a, b = st.columns(2)
            first_evaluations = a.number_input("Units evaluated", 0, 50, key="first_evaluations", step=1)
            first_approved = b.number_input("Units approved", 0, 50, key="first_approved", step=1)
        with s2:
            second_grade = st.slider("Average grade ", 0.0, 20.0, key="second_grade", step=0.1)
            a, b = st.columns(2)
            second_evaluations = a.number_input("Units evaluated ", 0, 50, key="second_evaluations", step=1)
            second_approved = b.number_input("Units approved ", 0, 50, key="second_approved", step=1)

    # ---------- validation ----------
    valid = True
    if first_approved > first_evaluations:
        st.error("1st semester: approved units can't exceed evaluated units.")
        valid = False
    if second_approved > second_evaluations:
        st.error("2nd semester: approved units can't exceed evaluated units.")
        valid = False

    # ---------- derived metrics ----------
    rate1 = first_approved / first_evaluations if first_evaluations else 0.0
    rate2 = second_approved / second_evaluations if second_evaluations else 0.0
    total_eval = first_evaluations + second_evaluations
    overall = (first_approved + second_approved) / total_eval if total_eval else 0.0
    trend = second_grade - first_grade

    k1, k2, k3, k4 = st.columns(4)
    for box, val, label in [
        (k1, f"{rate1:.0%}", "1st semester pass rate"),
        (k2, f"{rate2:.0%}", "2nd semester pass rate"),
        (k3, f"{overall:.0%}", "Overall pass rate"),
        (k4, f"{trend:+.1f}", "Grade change (sem 2 vs 1)"),
    ]:
        box.markdown(f'<div class="kpi"><div class="v">{val}</div><div class="l">{label}</div></div>',
                     unsafe_allow_html=True)

    st.write("")
    go = st.button("Predict student status", type="primary", disabled=not valid, use_container_width=True)

    if go:
        X = pd.DataFrame({
            "Age at enrollment": [age],
            "Admission grade": [admission_grade],
            "Course": [course],
            "Tuition fees up to date": [tuition],
            "Curricular units 2nd sem (grade)": [second_grade],
            "Curricular units 2nd sem (approved)": [second_approved],
            "Curricular units 2nd sem (evaluations)": [second_evaluations],
            "Curricular units 1st sem (evaluations)": [first_evaluations],
            "Curricular units 1st sem (approved)": [first_approved],
            "Curricular units 1st sem (grade)": [first_grade],
        })
        with st.spinner("Analyzing student profile..."):
            pred = int(model.predict(X)[0])
            probs = {}
            if hasattr(model, "predict_proba"):
                for c, p in zip(model.classes_, model.predict_proba(X)[0]):
                    probs[LABELS.get(int(c), str(c))] = float(p)
        result = LABELS.get(pred, str(pred))

        record = {"Time": dt.datetime.now().strftime("%H:%M:%S"), "Prediction": result,
                  "Confidence": f"{probs.get(result, 0):.0%}" if probs else "n/a",
                  **X.iloc[0].to_dict()}
        st.session_state.history.append(record)
        st.session_state.last = dict(result=result, probs=probs, inputs=X.iloc[0].to_dict(),
                                     rate1=rate1, rate2=rate2, trend=trend, tuition=tuition,
                                     grade1=first_grade, grade2=second_grade, overall=overall)

    # ---------- result ----------
    last = st.session_state.last
    if last:
        res, probs = last["result"], last["probs"]
        blurb = {
            "Dropout": "This profile resembles students who left before completing their course.",
            "Enrolled": "This profile resembles students who are still progressing through their course.",
            "Graduate": "This profile resembles students who completed their course successfully.",
        }.get(res, "")
        conf = f"{probs.get(res, 0):.0%} model confidence" if probs else ""
        st.markdown(
            f'<div class="result {res}"><div class="sm">Predicted status</div>'
            f'<p class="big">{res}</p><div class="sm">{blurb} {conf}</div></div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown("#### Probability breakdown")
            if probs:
                bars = ""
                for name in ["Dropout", "Enrolled", "Graduate"]:
                    p = probs.get(name, 0.0)
                    bars += (f'<div class="bar-row"><div class="name">{name}</div>'
                             f'<div class="bar-bg"><div class="bar-fill" style="width:{p*100:.1f}%;'
                             f'background:{COLORS[name]};"></div></div>'
                             f'<div class="pct">{p*100:.1f}%</div></div>')
                st.markdown(f'<div class="card">{bars}</div>', unsafe_allow_html=True)
            else:
                st.info("This model does not provide probabilities.")

        with right:
            st.markdown("#### Profile signals")
            chips = []
            chips.append(("good", "Fees up to date") if last["tuition"] == 1 else ("bad", "Fees not up to date"))
            chips.append(("good", "Strong pass rate") if last["overall"] >= 0.8
                         else ("warn", "Moderate pass rate") if last["overall"] >= 0.5
                         else ("bad", "Low pass rate"))
            chips.append(("good", "Grades improving") if last["trend"] > 0.5
                         else ("bad", "Grades declining") if last["trend"] < -0.5
                         else ("warn", "Grades stable"))
            chips.append(("good", "Average grade 12+") if min(last["grade1"], last["grade2"]) >= 12
                         else ("bad", "Average grade under 10") if min(last["grade1"], last["grade2"]) < 10
                         else ("warn", "Average grade 10-12"))
            html = "".join(f'<span class="chip {c}">{t}</span>' for c, t in chips)
            st.markdown(f'<div class="card">{html}<div style="font-size:12.5px;color:#5B6B7B;margin-top:8px;">'
                        'Quick checks calculated from the entered values. They are not model outputs.</div></div>',
                        unsafe_allow_html=True)

        st.markdown("#### Suggested next steps")
        steps = {
            "Dropout": ["Schedule an academic advising conversation soon.",
                        "Review tuition or financial-support options if fees are overdue.",
                        "Offer tutoring for units that were not approved."],
            "Enrolled": ["Keep monitoring pass rate across the next semester.",
                         "Encourage completion of any outstanding units.",
                         "Check in on workload and study habits."],
            "Graduate": ["Maintain current study habits and pace.",
                         "Consider peer-mentoring or enrichment opportunities.",
                         "Keep tuition and enrollment records up to date."],
        }[res]
        for s in steps:
            st.markdown(f"- {s}")

        report = (
            f"{APP_NAME} prediction report\nGenerated: {dt.datetime.now():%Y-%m-%d %H:%M}\n\n"
            f"Predicted status: {res}\n"
            + "".join(f"  {k}: {v:.1%}\n" for k, v in probs.items())
            + "\nInputs:\n" + "".join(f"  {k}: {v}\n" for k, v in last["inputs"].items())
        )
        d1, d2 = st.columns(2)
        d1.download_button("Download report (.txt)", report, f"{APP_NAME.lower()}_report.txt",
                           use_container_width=True)
        d2.download_button("Download inputs (.csv)", pd.DataFrame([last["inputs"]]).to_csv(index=False),
                           f"{APP_NAME.lower()}_inputs.csv", use_container_width=True)
    else:
        st.info("Fill in the details above, or load a sample profile from the sidebar, then select **Predict student status**.")

# ============================================================
# TAB 2 - INSIGHTS
# ============================================================
with tab_insights:
    st.subheader("Semester comparison")
    st.caption("Updates live as you change the form on the Predict tab.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Average grade (0-20)**")
        st.bar_chart(pd.DataFrame({"Grade": [first_grade, second_grade]},
                                  index=["1st semester", "2nd semester"]), color="#0F8B8D")
    with c2:
        st.markdown("**Units evaluated vs approved**")
        st.bar_chart(pd.DataFrame({"Evaluated": [first_evaluations, second_evaluations],
                                   "Approved": [first_approved, second_approved]},
                                  index=["1st semester", "2nd semester"]),
                     color=["#12263A", "#F2A541"])

    st.markdown("**Admission grade position (0-200)**")
    st.progress(min(admission_grade / 200, 1.0), text=f"{admission_grade:.1f} / 200")
    st.markdown("**Overall unit pass rate**")
    st.progress(min(overall, 1.0), text=f"{overall:.0%} of evaluated units approved")

# ============================================================
# TAB 3 - HISTORY
# ============================================================
with tab_history:
    st.subheader("Predictions this session")
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("Download history (.csv)", hist.to_csv(index=False),
                           f"{APP_NAME.lower()}_history.csv")
    else:
        st.info("No predictions yet. Run one on the Predict tab and it will appear here.")

# ============================================================
# TAB 4 - GUIDE
# ============================================================
with tab_guide:
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.subheader("How to use")
        st.markdown(
            "1. Load a sample profile or enter the student's data.\n"
            "2. Check that approved units don't exceed evaluated units.\n"
            "3. Select **Predict student status**.\n"
            "4. Review the probabilities, signals, and next steps.\n"
            "5. Download the report or compare runs in History."
        )
    with g2:
        st.subheader("Input reference")
        st.dataframe(pd.DataFrame({
            "Field": ["Age at enrollment", "Admission grade", "Course code", "Tuition fees",
                      "Semester grade", "Evaluated / approved units"],
            "Allowed values": ["15-100", "0-200", "Dataset course code", "Yes (1) / No (0)",
                               "0-20", "0-50, approved <= evaluated"],
        }), use_container_width=True, hide_index=True)

    st.subheader("About the model")
    st.write(
        f"{APP_NAME} uses a trained machine-learning classifier to estimate one of three outcomes: "
        "Dropout, Enrolled, or Graduate. Results reflect patterns in the training data and should "
        "support, not replace, decisions made by educators and advisors."
    )

st.divider()
st.caption(f"{APP_NAME} · {TAGLINE} · Machine Learning Demo")
