"""
TwinCore X - Public Teacher tool for detecting similarity in student assignment PDFs.
Supports multiple PDF upload. Ready for Streamlit Cloud / public deployment.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List

from utils import (
    extract_texts_from_uploaded_files,
    preprocess_texts,
    compute_embeddings,
    compute_similarity_matrix,
    get_suspicious_pairs,
    generate_pdf_report,
    generate_excel_report,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TwinCore X",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Authentication (simple for public demo)
# ---------------------------------------------------------------------------
HARDCODED_USERNAME = "teacher"
HARDCODED_PASSWORD = "twinx123"


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "logged_in": False,
        "texts": {},
        "sim_df": None,
        "suspicious_pairs": [],
        "filenames": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def login_page():
    """Render the login page."""
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h1>📄 TwinCore X</h1>
            <h3>Student Assignment Similarity Checker</h3>
            <p style="color: #666;">AI-powered semantic similarity detection for teachers</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("Login", use_container_width=True, type="primary")

            if submit:
                if username == HARDCODED_USERNAME and password == HARDCODED_PASSWORD:
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.info("**Demo Login**  \nUsername: `teacher`  \nPassword: `twinx123`")
        st.caption("This is a public demo. Do not upload sensitive student data.")


def logout():
    """Clear session and log out."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------
def render_sidebar():
    """Sidebar with logout and info."""
    with st.sidebar:
        st.markdown("### 📄 TwinCore X")
        st.markdown("---")
        st.markdown(f"**Logged in as:** `{HARDCODED_USERNAME}`")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        st.markdown("---")
        st.markdown(
            """
            **How to use**
            1. Upload multiple student PDF assignments
            2. Click **Analyze Assignments**
            3. Review the similarity heatmap & flagged pairs
            4. Select a pair for side-by-side comparison
            5. Download Evidence Report (PDF) or Excel
            """
        )
        st.markdown("---")
        st.markdown("**Threshold:** 75%")
        st.caption("Pairs above 75% similarity are flagged as suspicious.")
        st.markdown("---")
        st.caption("Built with Streamlit • PaddleOCR • sentence-transformers")


def upload_section():
    """Section for uploading multiple PDF files."""
    st.header("1. Upload Student Assignments")
    st.markdown(
        "Upload multiple student assignment PDFs (from Google Classroom, email, etc.). "
        "The system will extract text using OCR and compare them semantically."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple PDFs at once.",
    )

    if uploaded_files:
        st.success(f"**{len(uploaded_files)}** PDF file(s) selected")
        with st.expander("View uploaded files"):
            for f in uploaded_files:
                size_kb = len(f.getbuffer()) / 1024
                st.write(f"• {f.name} ({size_kb:.1f} KB)")

    analyze_btn = st.button(
        "🔍 Analyze Assignments",
        type="primary",
        use_container_width=False,
        disabled=not uploaded_files or len(uploaded_files) < 2,
    )

    if analyze_btn:
        if not uploaded_files or len(uploaded_files) < 2:
            st.error("Please upload at least **2 PDF files** to compare.")
            return

        with st.spinner("Running OCR + semantic similarity analysis... This may take 1–3 minutes depending on the number of files."):
            try:
                # 1. Extract text from uploaded files
                raw_texts = extract_texts_from_uploaded_files(uploaded_files)

                # 2. Preprocess
                cleaned = preprocess_texts(raw_texts)
                cleaned = {k: v for k, v in cleaned.items() if v.strip()}

                if len(cleaned) < 2:
                    st.error(
                        "Fewer than 2 PDFs produced usable text after OCR. "
                        "Try clearer scans or digital PDFs."
                    )
                    return

                # 3. Embeddings + Similarity
                filenames, embeddings = compute_embeddings(cleaned)
                sim_df = compute_similarity_matrix(filenames, embeddings)
                pairs = get_suspicious_pairs(sim_df, threshold=0.75)

                # Store results
                st.session_state.texts = cleaned
                st.session_state.sim_df = sim_df
                st.session_state.suspicious_pairs = pairs
                st.session_state.filenames = filenames

                st.success(
                    f"✅ Analysis complete! Analyzed **{len(cleaned)}** PDFs. "
                    f"Found **{len(pairs)}** suspicious pair(s) (≥ 75%)."
                )
                st.balloons()
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)


def results_section():
    """Display heatmap, stats and suspicious pairs table."""
    if st.session_state.sim_df is None:
        st.info("👆 Upload at least 2 PDFs and click **Analyze Assignments** to see results.")
        return

    st.header("2. Results")

    sim_df = st.session_state.sim_df
    pairs = st.session_state.suspicious_pairs
    n = len(st.session_state.filenames)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDFs Analyzed", n)
    c2.metric("Suspicious Pairs", len(pairs))
    max_sim = pairs[0][2] * 100 if pairs else 0.0
    c3.metric("Highest Similarity", f"{max_sim:.1f}%")
    avg_off_diag = 0.0
    if n > 1:
        vals = [sim_df.iloc[i, j] for i in range(n) for j in range(i + 1, n)]
        avg_off_diag = (sum(vals) / len(vals) * 100) if vals else 0
    c4.metric("Avg Pairwise Similarity", f"{avg_off_diag:.1f}%")

    st.subheader("Similarity Heatmap")
    fig = px.imshow(
        sim_df.values,
        x=sim_df.columns,
        y=sim_df.index,
        color_continuous_scale="RdYlGn_r",
        zmin=0,
        zmax=1,
        aspect="auto",
        labels=dict(color="Cosine Similarity"),
    )
    fig.update_layout(
        height=max(400, n * 38),
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_traces(
        hovertemplate="%{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Suspicious Pairs (≥ 75%)")
    if not pairs:
        st.success("🎉 No pairs exceeded the 75% similarity threshold.")
    else:
        pair_df = pd.DataFrame(
            [
                {
                    "Student A": a,
                    "Student B": b,
                    "Similarity (%)": round(score * 100, 2),
                }
                for a, b, score in pairs
            ]
        )
        st.dataframe(
            pair_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Similarity (%)": st.column_config.ProgressColumn(
                    "Similarity (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            },
        )


def comparison_section():
    """Side-by-side comparison of selected pair."""
    if not st.session_state.suspicious_pairs:
        return

    st.header("3. Side-by-Side Comparison")

    pairs = st.session_state.suspicious_pairs
    options = [f"{a}  ↔  {b}  ({score*100:.1f}%)" for a, b, score in pairs]
    selected = st.selectbox("Select a suspicious pair to compare", options)

    if selected:
        idx = options.index(selected)
        a, b, score = pairs[idx]
        text_a = st.session_state.texts.get(a, "(no text extracted)")
        text_b = st.session_state.texts.get(b, "(no text extracted)")

        st.markdown(f"**Similarity Score: {score*100:.1f}%**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 📄 {a}")
            st.text_area("Answer A", value=text_a, height=420, disabled=True, key=f"cmp_a_{idx}")
        with col2:
            st.markdown(f"#### 📄 {b}")
            st.text_area("Answer B", value=text_b, height=420, disabled=True, key=f"cmp_b_{idx}")


def report_section():
    """Download buttons for PDF and Excel reports."""
    if st.session_state.sim_df is None:
        return

    st.header("4. Download Evidence Report")

    pairs = st.session_state.suspicious_pairs
    texts = st.session_state.texts
    n = len(st.session_state.filenames)

    col1, col2 = st.columns(2)

    with col1:
        try:
            pdf_bytes = generate_pdf_report(
                total_pdfs=n,
                suspicious_pairs=pairs,
                texts=texts,
                threshold=0.75,
            )
            st.download_button(
                label="📥 Download Evidence Report (PDF)",
                data=pdf_bytes,
                file_name="TwinCoreX_Evidence_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")

    with col2:
        try:
            excel_bytes = generate_excel_report(pairs, threshold=0.75)
            st.download_button(
                label="📥 Export Suspicious Pairs (Excel)",
                data=excel_bytes,
                file_name="TwinCoreX_Suspicious_Pairs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Could not generate Excel: {e}")


def main_dashboard():
    """Main application after login."""
    render_sidebar()

    st.title("📄 TwinCore X Dashboard")
    st.markdown(
        "Detect **semantic similarity** across student assignment PDFs using OCR + AI embeddings. "
        "Useful as a screening tool for academic integrity."
    )
    st.markdown("---")

    upload_section()
    st.markdown("---")
    results_section()
    st.markdown("---")
    comparison_section()
    st.markdown("---")
    report_section()

    st.markdown("---")
    st.caption(
        "⚠️ High similarity is only a screening signal — it does **not** automatically prove plagiarism. "
        "Always perform manual review before taking action."
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    init_session_state()

    if not st.session_state.logged_in:
        login_page()
    else:
        main_dashboard()


if __name__ == "__main__":
    main()
