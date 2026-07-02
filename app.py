#!/usr/bin/env python3
import os
import uuid
from io import BytesIO
import re
import click
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for, flash, send_file, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func

# Try optional Firestore logger
try:
    from firestore_logger import FirestoreLogger
except Exception:
    FirestoreLogger = None

# AI helpers
from grad_cam import (
    load_keras_model, predict_with_model, generate_gradcam_or_dummy, ALLOWED_EXTENSIONS,
    LABELS
)
from symptoms_chat import run_symptom_checker

# -----------------
# Config
# -----------------
app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    f"sqlite:///{os.path.join(os.path.dirname(__file__), 'mediscan.db')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MODEL_PATH'] = os.environ.get('MODEL_PATH', os.path.join('models', 'model.h5'))
app.config['USE_FIRESTORE'] = os.environ.get('USE_FIRESTORE', 'false').lower() == 'true'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
db = SQLAlchemy(app)

# -----------------
# DB Models
# -----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now())

class Study(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ref_id = db.Column(db.String(36), unique=True, nullable=False)
    patient_name = db.Column(db.String(120))
    patient_age = db.Column(db.String(10))
    patient_sex = db.Column(db.String(10))
    note = db.Column(db.String(255))

    pre_image_path = db.Column(db.String(255))
    post_image_path = db.Column(db.String(255))
    pre_heatmap_path = db.Column(db.String(255))
    post_heatmap_path = db.Column(db.String(255))

    predicted_label = db.Column(db.String(120))
    confidence = db.Column(db.Float)
    findings_html = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=func.now())

    pdf_report_path = db.Column(db.String(255))

# -----------------
# Auth helpers
# -----------------
def current_user():
    uid = session.get('user_id')
    if not uid:
        return None
    return User.query.get(uid)

def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not current_user():
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)
    return wrapper

# -----------------
# CLI Commands
# -----------------
@app.cli.command("init-db")
def init_db_command():
    """Initialize the database (create all tables)."""
    with app.app_context():
        db.create_all()
    click.echo("✅ Database initialized.")

@app.cli.command("create-admin")
@click.argument("email")
@click.argument("password")
@click.option("--name", default="Admin", help="Name for the admin user")
def create_admin(email, password, name):
    """Create an admin/test user."""
    with app.app_context():
        if User.query.filter_by(email=email).first():
            click.echo("⚠️ User already exists.")
            return
        user = User(
            name=name,
            email=email.lower(),
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        click.echo(f"✅ Created user: {email}")

# -----------------
# Routes
# -----------------
@app.route('/')
def index():
    return render_template('index.html', user=current_user())

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', user=current_user())

@app.post('/api/chat')
def api_chat():
    """Chatbot API for symptoms checker."""
    data = request.get_json(force=True) or {}
    message = data.get('message', '').strip()

    if not message:
        return {"reply": "⚠️ Please provide a message."}, 400

    try:
        reply = run_symptom_checker(message)
    except Exception as e:
        print("Chatbot error:", e)
        reply = "⚠️ Sorry, the chatbot is temporarily unavailable."

    return {"reply": reply}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))
        session['user_id'] = user.id
        flash('Welcome back!', 'success')
        return redirect(url_for('xray'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not name or not email or not password:
            flash('Please fill all fields.', 'warning')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'warning')
            return redirect(url_for('register'))
        user = User(name=name, email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        flash('Account created. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.get('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/xray')
@login_required
def xray():
    return render_template('xray.html', user=current_user())

# -----------------
# X-ray Upload + Analysis
# -----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post('/xray/analyze')
@login_required
def analyze():
    patient_name = request.form.get('patient_name', '')
    patient_age = request.form.get('patient_age', '')
    patient_sex = request.form.get('patient_sex', '')
    note = request.form.get('note', '')

    pre_file = request.files.get('pre_image')
    post_file = request.files.get('post_image')

    if not pre_file or pre_file.filename == '' or not allowed_file(pre_file.filename):
        flash('Please upload at least a PRE image (jpg/png).', 'danger')
        return redirect(url_for('xray'))

    ref_id = str(uuid.uuid4())
    study_dir = os.path.join(app.config['UPLOAD_FOLDER'], ref_id)
    os.makedirs(study_dir, exist_ok=True)

    def save_uploaded(fileobj, name_hint):
        fname = secure_filename(fileobj.filename)
        ext = os.path.splitext(fname)[1].lower() or '.png'
        final = f"{name_hint}{ext}"
        path = os.path.join(study_dir, final)
        fileobj.save(path)
        return path

    pre_path = save_uploaded(pre_file, 'pre_original')
    post_path = None
    if post_file and post_file.filename and allowed_file(post_file.filename):
        post_path = save_uploaded(post_file, 'post_original')

    model = load_keras_model(app.config['MODEL_PATH'])
    pre_pred, pre_conf = predict_with_model(model, pre_path)
    pre_heatmap_path = generate_gradcam_or_dummy(model, pre_path, os.path.join(study_dir, 'pre_heatmap.png'))

    post_pred, post_conf, post_heatmap_path = None, None, None
    if post_path:
        post_pred, post_conf = predict_with_model(model, post_path)
        post_heatmap_path = generate_gradcam_or_dummy(model, post_path, os.path.join(study_dir, 'post_heatmap.png'))

    final_label = pre_pred or 'Unknown'
    final_conf = pre_conf
    display_label = 'Normal / Unremarkable' if final_label.lower() in ['no finding', 'no_finding', 'nofinding'] else final_label

    findings_html = f"<p><strong>AI Findings:</strong> {display_label}"
    if final_conf is not None:
        findings_html += f" (confidence: {final_conf:.2%})"
    findings_html += "</p>"

    study = Study(
        ref_id=ref_id,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_sex=patient_sex,
        note=note,
        pre_image_path=pre_path,
        post_image_path=post_path,
        pre_heatmap_path=pre_heatmap_path,
        post_heatmap_path=post_heatmap_path,
        predicted_label=final_label,
        confidence=final_conf,
        findings_html=findings_html
    )
    db.session.add(study)
    db.session.commit()

    if app.config['USE_FIRESTORE'] and FirestoreLogger is not None:
        try:
            FirestoreLogger().log_prediction(
                ref_id=ref_id,
                image_name=os.path.basename(pre_path),
                label=final_label,
                confidence=float(final_conf or 0.0)
            )
        except Exception as e:
            print("Firestore logging failed:", e)

    flash('Analysis complete. You can edit findings and save.', 'success')
    return redirect(url_for('record', ref_id=ref_id))

# -----------------
# Record + PDF Report
# -----------------
@app.route('/record/<ref_id>', methods=['GET', 'POST'])
@login_required
def record(ref_id):
    study = Study.query.filter_by(ref_id=ref_id).first_or_404()
    if request.method == 'POST':
        study.findings_html = request.form.get('findings_html', '')
        db.session.commit()
        flash('Findings saved to database.', 'success')
        return redirect(url_for('record', ref_id=ref_id))
    return render_template('record.html', study=study, user=current_user())

from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Frame

@app.get('/report/<ref_id>.pdf')
@login_required
def report_pdf(ref_id):
    study = Study.query.filter_by(ref_id=ref_id).first_or_404()
    buffer = BytesIO()
    c = pdf_canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "MediScan AI - X-ray Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Reference ID: {study.ref_id}")
    c.drawString(50, height - 110, f"Patient: {study.patient_name or '-'}")
    c.drawString(50, height - 130, f"Age/Sex: {study.patient_age or '-'} / {study.patient_sex or '-'}")
    c.drawString(50, height - 150, f"Timestamp: {study.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    label = study.predicted_label or 'Unknown'
    if label.lower() in ['no finding', 'no_finding', 'nofinding']:
        label = 'Normal / Unremarkable'
    conf_str = f"{(study.confidence or 0.0):.2%}"
    c.drawString(50, height - 180, f"AI Prediction: {label} (confidence: {conf_str})")

    y_img = height - 420
    try:
        if study.pre_image_path and os.path.exists(study.pre_image_path):
            c.drawString(50, y_img + 220, "Original (pre)")
            c.drawImage(ImageReader(study.pre_image_path), 50, y_img, width=220, height=220, preserveAspectRatio=True, mask='auto')
        if study.pre_heatmap_path and os.path.exists(study.pre_heatmap_path):
            c.drawString(300, y_img + 220, "Grad-CAM (pre)")
            c.drawImage(ImageReader(study.pre_heatmap_path), 300, y_img, width=220, height=220, preserveAspectRatio=True, mask='auto')
    except Exception as e:
        print("PDF image embedding error:", e)

    text = re.sub('<[^<]+?>', '', study.findings_html or '')
    c.drawString(50, y_img - 20, "Findings:")
    stylesheet = getSampleStyleSheet()
    frame = Frame(50, 80, width - 100, y_img - 120, showBoundary=0)
    story = [Paragraph(text.replace('\n', '<br/>'), stylesheet['BodyText'])]
    frame.addFromList(story, c)

    c.showPage()
    c.save()
    buffer.seek(0)

    report_folder = os.path.join(app.config['UPLOAD_FOLDER'], ref_id)
    os.makedirs(report_folder, exist_ok=True)
    pdf_path = os.path.join(report_folder, 'report.pdf')
    with open(pdf_path, 'wb') as f:
        f.write(buffer.getvalue())
    study.pdf_report_path = pdf_path
    db.session.commit()

    return send_file(buffer, mimetype='application/pdf', as_attachment=True,
                     download_name=f"MediScanAI_{ref_id}.pdf")

# -----------------
# Utilities
# -----------------
@app.context_processor
def inject_globals():
    return dict(LABELS=LABELS)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
