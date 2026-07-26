"""
CV Analyzer Dashboard
----------------------
Upload a Job Description + multiple CVs, score each candidate against the JD
on 7 dimensions using an LLM (via OpenRouter), and explore results in an
interactive Streamlit dashboard with a side-by-side CV / scorecard view.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import json
import re

import fitz  # PyMuPDF
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIG / CONSTANTS
# ─────────────────────────────────────────────
st.set_page_config(page_title="CV Analyzer Dashboard", layout="wide")

DIMENSIONS = [
    "education",
    "work_experience",
    "skills",
    "certifications_and_courses",
    "languages",
    "technology_knowledge",
    "general_assessment",
]

CANDIDATE_STRUCTURE = {
    "full_name": "", "email": "", "phone": "", "location": "",
    "latest_qualification": "", "last_school_or_university": "",
    "graduation_year": "", "latest_job_title": "", "latest_employer": "",
    "latest_job_duration": "", "years_of_experience": "",
    "top_skills": [], "certifications": [], "languages": [], "systems_known": [],
}

POSITION_STRUCTURE = {
    "JobTitle": "", "Department": "", "ReportsTo": "", "JobPurpose": "",
    "KeyResponsibilities": [], "WorkModality": "", "AdditionalDetails": "",
    "Qualifications": {
        "education": [], "work_experience": [], "skills": [],
        "certifications_and_courses": [],
        "languages": [{"language": "", "proficiency": ""}],
        "systems_knowledge": [],
    },
}

EVALUATION_STRUCTURE = {
    "dimensions": [{"name": d, "value": 0, "reasoning": ""} for d in DIMENSIONS]
}

MODEL_OPTIONS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3.1-70b-instruct",
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    doc = fitz.open(stream=data, filetype="pdf")
    return "".join(page.get_text() for page in doc)


def extract_json(raw: str) -> dict:
    """Robustly pull a JSON object out of an LLM response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def call_llm(client, model, system_prompt, user_prompt) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def normalize_jd(client, model, jd_text):
    raw = call_llm(
        client, model,
        "You are a recruiter assistant. Convert job descriptions into structured JSON. Return ONLY valid JSON, no explanation.",
        f"Extract and return this job description as JSON using exactly this structure:\n\n{json.dumps(POSITION_STRUCTURE, indent=2)}\n\nJob Description:\n{jd_text}",
    )
    return extract_json(raw)


def extract_profile(client, model, cv_text):
    raw = call_llm(
        client, model,
        "You are a CV parser. Extract candidate information into structured JSON. Return ONLY valid JSON, no explanation.",
        f"Extract candidate details from this CV using exactly this JSON structure:\n\n{json.dumps(CANDIDATE_STRUCTURE, indent=2)}\n\nCV:\n{cv_text}",
    )
    return extract_json(raw)


def evaluate_cv(client, model, structured_jd, cv_text):
    raw = call_llm(
        client, model,
        "You are a professional recruiter. Evaluate candidates against job descriptions. Return ONLY valid JSON, no explanation.",
        f"""Evaluate this candidate's CV against the job description.
Use this exact JSON structure (scores 0-10):
{json.dumps(EVALUATION_STRUCTURE, indent=2)}

Job Description (structured):
{json.dumps(structured_jd, indent=2)}

Candidate CV:
{cv_text}

Return valid JSON only.""",
    )
    return extract_json(raw)


def score_color(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = 0
    if v >= 7:
        return "#C6EFCE", "#0B6B2B"   # bg, text
    elif v >= 4:
        return "#FFEB9C", "#7A5B00"
    return "#FFC7CE", "#8A1F1F"


def score_badge_html(name, value, reasoning):
    bg, fg = score_color(value)
    label = name.replace("_", " ").title()
    return f"""
    <div style="border:1px solid #e0e0e0; border-radius:10px; padding:12px 16px; margin-bottom:10px; background:#fafafa;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:600; font-size:0.95rem;">{label}</span>
            <span style="background:{bg}; color:{fg}; font-weight:700; padding:2px 12px; border-radius:20px; font-size:0.9rem;">
                {value}/10
            </span>
        </div>
        <div style="margin-top:6px; color:#444; font-size:0.87rem; line-height:1.4;">{reasoning}</div>
    </div>
    """


def build_excel(results_df: pd.DataFrame) -> bytes:
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Evaluation Results")
        ws = writer.sheets["Evaluation Results"]

        header_fill = PatternFill("solid", fgColor="1F3864")
        header_font = Font(color="FFFFFF", bold=True, size=11)
        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        thin_border = Border(left=Side(style="thin"), right=Side(style="thin"),
                              top=Side(style="thin"), bottom=Side(style="thin"))

        for cell in ws[1]:
            cell.fill, cell.font, cell.alignment, cell.border = header_fill, header_font, center_align, thin_border

        score_cols = [c for c in results_df.columns if c.endswith("_score")]
        score_col_idx = [results_df.columns.get_loc(c) + 1 for c in score_cols]

        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="center")
                cell.border = thin_border
                if cell.column in score_col_idx:
                    try:
                        v = float(cell.value or 0)
                        bg, _ = score_color(v)
                        cell.fill = PatternFill("solid", fgColor=bg.replace("#", ""))
                    except Exception:
                        pass

        for col_num in range(1, ws.max_column + 1):
            col_letter = get_column_letter(col_num)
            max_len = max((len(str(ws.cell(row=r, column=col_num).value or ""))
                           for r in range(1, ws.max_row + 1)), default=10)
            ws.column_dimensions[col_letter].width = min(max_len + 4, 45)
        ws.freeze_panes = "A2"

    return buf.getvalue()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []       # list of per-candidate dicts
if "jd_json" not in st.session_state:
    st.session_state.jd_json = None
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    secret_key = st.secrets.get("OPENROUTER_API_KEY", "") if hasattr(st, "secrets") else ""
    if secret_key:
        api_key = secret_key
        st.success("API key loaded from secrets ✅")
    else:
        api_key = st.text_input("OpenRouter API Key", type="password",
                                 help="Get one at openrouter.ai. Never hardcode this in source files. "
                                      "To avoid typing it every time, add it to .streamlit/secrets.toml "
                                      "as OPENROUTER_API_KEY when deploying.")

    default_model_idx = MODEL_OPTIONS.index(st.secrets.get("OPENROUTER_MODEL")) \
        if hasattr(st, "secrets") and st.secrets.get("OPENROUTER_MODEL") in MODEL_OPTIONS else 0
    model = st.selectbox("Model", MODEL_OPTIONS, index=default_model_idx)

    st.markdown("---")
    jd_file = st.file_uploader("Job Description (PDF)", type="pdf")
    cv_files = st.file_uploader("Candidate CVs (PDF)", type="pdf", accept_multiple_files=True)

    run = st.button("🚀 Run Analysis", type="primary", use_container_width=True,
                     disabled=not (api_key and jd_file and cv_files))
    if not api_key:
        st.caption("Enter your API key to enable analysis.")

# ─────────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────────
if run:
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    st.session_state.results = []

    with st.spinner("Reading job description..."):
        jd_text = extract_pdf_text(jd_file)
        st.session_state.jd_text = jd_text
        try:
            st.session_state.jd_json = normalize_jd(client, model, jd_text)
        except Exception as e:
            st.error(f"Failed to normalize JD: {e}")
            st.session_state.jd_json = POSITION_STRUCTURE

    progress = st.progress(0.0, text="Starting candidate analysis...")
    n = len(cv_files)
    for i, cv_file in enumerate(cv_files):
        progress.progress((i) / n, text=f"Analyzing {cv_file.name}...")
        cv_text = extract_pdf_text(cv_file)

        try:
            profile = extract_profile(client, model, cv_text)
        except Exception as e:
            st.warning(f"Profile extraction failed for {cv_file.name}: {e}")
            profile = CANDIDATE_STRUCTURE.copy()

        try:
            eval_data = evaluate_cv(client, model, st.session_state.jd_json, cv_text)
        except Exception as e:
            st.warning(f"Evaluation failed for {cv_file.name}: {e}")
            eval_data = EVALUATION_STRUCTURE.copy()

        dims = {d["name"]: d for d in eval_data.get("dimensions", [])}
        scores = [dims.get(d, {}).get("value", 0) for d in DIMENSIONS]
        overall = round(sum(float(s) for s in scores) / len(scores), 2) if scores else 0

        st.session_state.results.append({
            "filename": cv_file.name,
            "cv_text": cv_text,
            "profile": profile,
            "dimensions": dims,
            "overall_score": overall,
        })

    progress.progress(1.0, text="Done!")
    st.success(f"Analyzed {n} candidate(s).")

# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
st.title("📋 CV Analyzer Dashboard")

if not st.session_state.results:
    st.info("Upload a JD + CVs in the sidebar and click **Run Analysis** to get started.")
    st.stop()

results = st.session_state.results
df_rows = []
for r in results:
    row = {"Candidate": r["profile"].get("full_name") or r["filename"], "Filename": r["filename"],
           "Overall Score": r["overall_score"]}
    for d in DIMENSIONS:
        row[d.replace("_", " ").title()] = r["dimensions"].get(d, {}).get("value", 0)
    df_rows.append(row)

df = pd.DataFrame(df_rows).sort_values("Overall Score", ascending=False).reset_index(drop=True)
df.insert(0, "Rank", df.index + 1)

tab_overview, tab_detail = st.tabs(["📊 Overview", "🔍 Candidate Detail"])

# ---------- OVERVIEW TAB ----------
with tab_overview:
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("Ranked Results")
        score_cols = [c for c in df.columns if c not in ("Rank", "Candidate", "Filename")]
        styled = df.style.background_gradient(subset=score_cols, cmap="RdYlGn", vmin=0, vmax=10)
        st.dataframe(styled, use_container_width=True, height=min(60 + 40 * len(df), 500))

        excel_bytes = build_excel(df.drop(columns=["Filename"]))
        st.download_button("⬇️ Download Excel Report", data=excel_bytes,
                            file_name="CV_Evaluation_Results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col2:
        st.subheader("Overall Score by Candidate")
        fig_bar = px.bar(df, x="Overall Score", y="Candidate", orientation="h",
                          color="Overall Score", color_continuous_scale="RdYlGn",
                          range_color=[0, 10])
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Dimension Heatmap Across Candidates")
    dim_cols = [d.replace("_", " ").title() for d in DIMENSIONS]
    heat_df = df.set_index("Candidate")[dim_cols]
    fig_heat = px.imshow(heat_df, text_auto=True, aspect="auto",
                          color_continuous_scale="RdYlGn", zmin=0, zmax=10)
    fig_heat.update_layout(height=120 + 40 * len(df))
    st.plotly_chart(fig_heat, use_container_width=True)

# ---------- DETAIL TAB ----------
with tab_detail:
    names = [r["profile"].get("full_name") or r["filename"] for r in results]
    choice = st.selectbox("Select candidate", names)
    candidate = next(r for r, n in zip(results, names) if n == choice)
    profile = candidate["profile"]
    dims = candidate["dimensions"]

    st.markdown(f"### {choice}  —  Overall Score: **{candidate['overall_score']}/10**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Email", profile.get("email") or "—")
    c2.metric("Location", profile.get("location") or "—")
    c3.metric("Latest Role", profile.get("latest_job_title") or "—")
    c4.metric("Experience", str(profile.get("years_of_experience") or "—"))

    # Radar chart
    radar_vals = [float(dims.get(d, {}).get("value", 0)) for d in DIMENSIONS]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=radar_vals + [radar_vals[0]],
                                         theta=[d.replace("_", " ").title() for d in DIMENSIONS] + [DIMENSIONS[0].replace("_", " ").title()],
                                         fill="toself", name=choice))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                             showlegend=False, height=380, margin=dict(t=30, b=30))
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.subheader("Side-by-Side: CV vs. AI Scoring")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("**📄 CV Text**")
        st.markdown(
            f"""<div style="height:650px; overflow-y:auto; border:1px solid #ddd; border-radius:8px;
                 padding:14px; background:#fff; font-size:0.85rem; white-space:pre-wrap; line-height:1.5;">
                 {candidate['cv_text']}</div>""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("**🧠 AI Scorecard & Interpretation**")
        badges_html = "".join(
            score_badge_html(d, dims.get(d, {}).get("value", 0), dims.get(d, {}).get("reasoning", "No reasoning provided."))
            for d in DIMENSIONS
        )
        st.markdown(f'<div style="height:650px; overflow-y:auto; padding-right:6px;">{badges_html}</div>',
                     unsafe_allow_html=True)
