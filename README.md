# TwinCore X (Public Version)

**AI-powered teacher tool to detect semantic similarity between student assignment PDFs.**

Upload multiple student PDFs → OCR → Semantic embeddings → Similarity heatmap → Flag suspicious pairs → Side-by-side comparison → Download evidence report.

Perfect for public deployment on **Streamlit Community Cloud** or any server.

---

## Features

- Simple login (demo credentials included)
- **Multiple PDF upload** (no local folder path needed)
- Text extraction with **PaddleOCR**
- Semantic similarity using **sentence-transformers**
- Interactive Plotly heatmap
- Automatic flagging of pairs ≥ 75% similarity
- Side-by-side answer comparison
- Downloadable **Evidence Report (PDF)** + **Excel export**

---

## Demo Login

| Field     | Value       |
|-----------|-------------|
| Username  | `teacher`   |
| Password  | `twinx123`  |

---

## Quick Start (Local)

```bash
cd TwinCoreX

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud (Public Link)

1. Create a free account at https://share.streamlit.io
2. Push this project to a **public GitHub repository**
3. Click **"New app"** on Streamlit Cloud
4. Select your repository → main branch → `app.py`
5. Click **Deploy**

Your public link will look like:  
`https://your-app-name.streamlit.app`

> **Note**: First run may take longer while models download.  
> Streamlit Cloud free tier has memory limits. For many large PDFs, a paid plan or VPS is better.

---

## Deploy on Hugging Face Spaces

1. Create a new Space → Select **Streamlit**
2. Upload all files (`app.py`, `utils.py`, `requirements.txt`)
3. Wait for build

---

## Project Structure

```
TwinCoreX/
├── app.py              # Main Streamlit app (upload-based)
├── utils.py            # OCR, embeddings, reports
├── requirements.txt
└── README.md
```

---

## Important Notes

- High similarity ≠ confirmed plagiarism. Always do manual review.
- Works best with clear digital PDFs or good-quality scans.
- Do **not** upload real student data containing personal information on public demos.
- Models are downloaded on first run (~300-500 MB total).

---

## License

For educational and teacher use. Not for fully automated high-stakes decisions.
