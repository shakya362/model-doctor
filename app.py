import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from model_doctor import audit
from model_doctor.report import to_html

st.set_page_config(page_title="Model Doctor", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #f7f8fa; }
    .verdict-badge {
        display:inline-block; padding:4px 14px; border-radius:999px;
        font-weight:600; font-size:0.85rem; letter-spacing:.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 Model Doctor")
st.caption("Upload a dataset (+ optional trained model / training script) and get a client-readable ML audit.")

with st.sidebar:
    st.header("Inputs")
    data_file = st.file_uploader("Dataset (CSV)", type=["csv"])
    model_file = st.file_uploader("Trained model (.pkl / .joblib) — optional", type=["pkl", "joblib"])
    code_file = st.file_uploader("Training script (.py / .ipynb) — optional", type=["py", "ipynb"])

    target = None
    if data_file is not None:
        preview_df = pd.read_csv(data_file)
        data_file.seek(0)
        target = st.selectbox("Target column", preview_df.columns.tolist())

    goal = st.text_input("Stated goal (optional)", placeholder="e.g. Predict which customers will churn")
    task = st.selectbox("Task override (optional)", ["auto", "classification", "regression"])
    time_column = st.text_input("Time column (optional)")
    group_column = st.text_input("Group/entity column (optional)")
    run_autofix = st.checkbox("Run autofix (before/after re-measure)", value=True)

    run_btn = st.button("Run audit", type="primary", use_container_width=True, disabled=data_file is None)

if run_btn and data_file is not None and target:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_path = tmp / "data.csv"
        data_path.write_bytes(data_file.getvalue())

        model_path = None
        if model_file is not None:
            model_path = tmp / model_file.name
            model_path.write_bytes(model_file.getvalue())

        code_path = None
        if code_file is not None:
            code_path = tmp / code_file.name
            code_path.write_bytes(code_file.getvalue())

        with st.spinner("Running checks..."):
            result = audit(
                data=data_path,
                target=target,
                model=str(model_path) if model_path else None,
                code=str(code_path) if code_path else None,
                goal=goal or None,
                task=None if task == "auto" else task,
                time_column=time_column or None,
                group_column=group_column or None,
                autofix=run_autofix,
            )

        color = {"Healthy": "#1a7f37", "Needs attention": "#b78103", "Not safe to deploy": "#cf222e"}.get(
            result.verdict, "#57606a"
        )
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(
                f'<span class="verdict-badge" style="background:{color}22;color:{color};">{result.verdict}</span>',
                unsafe_allow_html=True,
            )
        with c2:
            st.metric("Health score", f"{result.health_score:.0f}/100")
        with c3:
            counts = {}
            for f in result.findings:
                counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
            st.write(" · ".join(f"{k}: {v}" for k, v in counts.items()) or "No findings")

        tab1, tab2 = st.tabs(["Report", "Findings table"])

        with tab1:
            html = to_html(result)
            st.components.v1.html(html, height=1000, scrolling=True)
            st.download_button(
                "Download full HTML report",
                data=html,
                file_name="model_doctor_report.html",
                mime="text/html",
            )

        with tab2:
            rows = [
                {
                    "check_id": f.check_id,
                    "severity": f.severity.value,
                    "confidence": round(f.confidence, 2),
                    "title": f.title,
                    "auto_fixable": f.auto_fixable,
                }
                for f in result.findings
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif run_btn and not target:
    st.warning("Upload a dataset and pick a target column first.")
else:
    st.info("Upload a dataset from the sidebar to begin. Model and training script are optional.")
