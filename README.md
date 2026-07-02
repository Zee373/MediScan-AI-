# MediScan AI – Flask Frontend (Red/Black/White)

A production-ready Flask frontend for your MediScan AI project with:

- ✅ Home with two-button UI (Symptoms vs X‑ray) – matching your screenshots
- ✅ Doctor auth (Register/Login/Logout)
- ✅ X‑ray page: upload pre/post images, generate Grad‑CAM (or dummy heatmap), view side-by-side
- ✅ Editable rich‑text "Findings" stored in DB
- ✅ PDF report generation + image download links
- ✅ API route for Symptoms Checker chatbot
- ✅ SQLite storage (optional Firestore logging)
- ✅ Works even if TensorFlow/model is missing (falls back to dummy heatmap)

> **Color theme:** strict red/black/white across all pages.

---

## 1) Project Structure

```
mediscan_ai/
├── app.py
├── grad_cam.py
├── symptoms_chat.py
├── firestore_logger.py        # optional Firestore
├── mediscan.db                # auto-created on first run
├── models/
│   └── model.h5               # <— put your trained model here
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/               # auto-created; original + heatmaps stored per ref_id
└── templates/
    ├── base.html
    ├── index.html
    ├── chatbot.html
    ├── xray.html
    ├── login.html
    ├── register.html
    └── record.html
```

---

## 2) Setup (VS Code)

```bash
# Clone/copy this folder
cd mediscan_ai

# 1) Create venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Put your trained model
# Copy your Jupyter-trained model file into:
#   models/model.h5
# (If missing, app still runs with a dummy heatmap.)

# 4) Initialize DB (first run creates tables automatically)
flask --app app.py init-db

# 5) Run
flask --app app.py run  # or: python app.py
```

Open http://127.0.0.1:5000

**Doctor access:** Register → Login → X‑ray page.

---

## 3) How Jupyter ↔ Flask connect

1. In Jupyter, export your trained Keras model as `model.h5`.
2. Copy `model.h5` into the Flask project at `models/model.h5`.
3. The app loads it via `app.config['MODEL_PATH']`.  
   If not found or TensorFlow is not installed on the server, the app still works using a dummy Grad‑CAM so you can demo the UI.

If you kept additional preprocessing steps in the notebook, replicate them in `grad_cam.py` inside `_prepare_image_for_model()`. That function is the single place to standardize image size/normalization.

---

## 4) Database Schema

SQLite via SQLAlchemy with two tables:

**User**
- id (PK), email (unique), name, password_hash, created_at

**Study**
- id (PK)
- ref_id (UUID string, unique)
- patient_name, patient_age, patient_sex, note
- pre_image_path, post_image_path
- pre_heatmap_path, post_heatmap_path
- predicted_label, confidence
- findings_html (Rich Text, editable by doctor)
- timestamp, pdf_report_path

> "No Finding" is presented as **Normal / Unremarkable** in the UI & PDF.

---

## 5) Firestore (optional)

If you prefer Firestore logging (Week 4 optional):
1. `pip install google-cloud-firestore`
2. Set env vars:
   - `USE_FIRESTORE=true`
   - Provide application default credentials (`GOOGLE_APPLICATION_CREDENTIALS=/path/key.json`)
3. The app will write `{ref_id, image_name, label, confidence, timestamp}` to Firestore.

---

## 6) PDF Reports & Downloads

- Route: `/report/<ref_id>.pdf`
- Includes patient info, timestamps, AI prediction + confidence, pre image + Grad‑CAM, and findings.
- Original and heatmap images have **Download** buttons on the record page.

---

## 7) Symptoms Checker API

- Frontend: `/chatbot` (simple chat UI)
- Backend: `/api/chat`
- Replace demo logic inside `symptoms_chat.py::run_symptom_checker()` with your real model/parser (import your `.py` file from the notebook).

---

## 8) Streamlit (optional)

You asked about Streamlit. You can reuse `grad_cam.py` in a `streamlit_app.py` for a standalone demo. This repo focuses on Flask for integration with auth/DB. If you still want a Streamlit panel, create a new file and import:
```python
from grad_cam import load_keras_model, predict_with_model, generate_gradcam_or_dummy
```
and mirror the upload → predict → heatmap → download flow.

---

## 9) Deployment Roadmap

**Backend (Render or similar):**
- Deploy this Flask app. Ensure TensorFlow is supported or keep dummy heatmap.

**Frontend (Vercel)**
- You can deploy a thin static layer (e.g., a landing page), but this app already serves HTML. The simplest option is to deploy the Flask app only.

**Database**
- SQLite for demo.
- Switch to Postgres (Render) or Firestore by updating `SQLALCHEMY_DATABASE_URI` / setting `USE_FIRESTORE=true`.

**Final Demo Check**
- Chatbot ↔ API: `/api/chat`
- X‑ray upload → predict → heatmap → save → report
- DB rows created and retrievable via `/record/<ref_id>`.

---

## 10) Notes

- Allowed image types: JPG/PNG.
- Input size defaults to 224×224; adjust in `grad_cam.py` if your model uses a different size.
- If your classes order differs, update `LABELS` in `grad_cam.py` accordingly.
- Security/medical disclaimer: This UI is for **demo & research only**, not for clinical use.
```

