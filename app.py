# app.py  –  AI Resume Analyzer
import os, re, uuid, json, sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from docx import Document as DocxDoc
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    NLP_OK = True
except Exception:
    NLP_OK = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "resumeai-secret-2024"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
ALLOWED = {"pdf", "docx"}
DB = "resume_analyzer.db"

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY, filename TEXT, job_title TEXT,
            score INTEGER, matched TEXT, missing TEXT,
            suggestions TEXT, keywords INTEGER, created_at TEXT
        )
    """)
    con.commit(); con.close()

def db_save(rec):
    con = sqlite3.connect(DB)
    con.execute("""INSERT INTO analyses
        (id,filename,job_title,score,matched,missing,suggestions,keywords,created_at)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (rec["id"], rec["filename"], rec["job_title"], rec["score"],
         json.dumps(rec["matched"]), json.dumps(rec["missing"]),
         json.dumps(rec["suggestions"]), rec["keywords"], rec["created_at"]))
    con.commit(); con.close()

def db_all():
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM analyses ORDER BY created_at DESC LIMIT 30")]
    con.close()
    for r in rows:
        r["matched"] = json.loads(r["matched"])
        r["missing"] = json.loads(r["missing"])
        r["suggestions"] = json.loads(r["suggestions"])
    return rows

def db_get(aid):
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    con.close()
    if not row: return None
    r = dict(row)
    r["matched"] = json.loads(r["matched"])
    r["missing"] = json.loads(r["missing"])
    r["suggestions"] = json.loads(r["suggestions"])
    return r

def db_delete(aid):
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM analyses WHERE id=?", (aid,))
    con.commit(); con.close()

# ── File utilities ────────────────────────────────────────────────────────────
def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def extract_text(path):
    ext = path.rsplit(".", 1)[1].lower()
    text = ""
    if ext == "pdf" and PDF_OK:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    elif ext == "docx" and DOCX_OK:
        doc = DocxDoc(path)
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        try:
            with open(path, "r", errors="ignore") as f:
                text = f.read()
        except Exception:
            pass
    return text.strip()

# ── NLP / AI Module ───────────────────────────────────────────────────────────
SKILLS = [
    "python","java","javascript","typescript","c++","c#","go","rust","php","ruby",
    "react","angular","vue","next.js","html","css","tailwind","bootstrap",
    "node.js","flask","django","fastapi","spring","express",
    "sql","mysql","postgresql","mongodb","sqlite","redis","firebase",
    "docker","kubernetes","aws","azure","gcp","terraform","linux","git","ci/cd",
    "machine learning","deep learning","nlp","computer vision","data analysis",
    "scikit-learn","tensorflow","pytorch","pandas","numpy","matplotlib",
    "rest api","graphql","microservices","agile","scrum","devops",
    "figma","photoshop","excel","powerbi","tableau","communication",
    "leadership","teamwork","problem solving","project management",
    "selenium","pytest","junit","testing","cybersecurity","networking",
]

def extract_skills(text):
    t = text.lower()
    return [s for s in SKILLS if re.search(r"\b" + re.escape(s) + r"\b", t)]

def extract_education(text):
    degrees = ["bachelor","master","phd","doctorate","diploma","associate",
               "b.sc","m.sc","mba","b.eng","m.eng"]
    found = []
    for line in text.split("\n"):
        if any(d in line.lower() for d in degrees) and line.strip():
            found.append(line.strip())
    return found[:5]

def extract_experience(text):
    patterns = [r"\d+\s*\+?\s*year", r"experience", r"worked at",
                r"internship", r"position", r"developed", r"managed"]
    exp = []
    for line in text.split("\n"):
        if any(re.search(p, line.lower()) for p in patterns) and len(line.strip()) > 15:
            exp.append(line.strip())
    return exp[:6]

def similarity_score(resume_text, job_text):
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        mat = vec.fit_transform([resume_text, job_text])
        return min(round(cosine_similarity(mat[0:1], mat[1:2])[0][0] * 100), 99)
    except Exception:
        return 0

def make_suggestions(score, matched, missing):
    tips = []
    if score >= 75:
        tips.append({"type":"ok","title":"Strong overall match",
            "text":"You are a great fit. Quantify achievements with numbers to stand out even more."})
    elif score >= 50:
        tips.append({"type":"info","title":"Moderate match",
            "text":"You cover the basics. Adding the missing skills below will boost your score significantly."})
    else:
        tips.append({"type":"warn","title":"Low match — tailor your resume",
            "text":"Rewrite your resume to mirror the language and keywords in the job description."})
    if missing:
        tips.append({"type":"warn","title":"Add missing keywords",
            "text":"These appear in the job description but not your resume: " + ", ".join(missing[:5]) + "."})
    if len(matched) >= 6:
        tips.append({"type":"ok","title":"Good skills coverage",
            "text":f"You matched {len(matched)} skills. Make each one visible in your experience bullets."})
    else:
        tips.append({"type":"info","title":"Expand your skills section",
            "text":"List all relevant tools, languages, and methodologies — recruiters use keyword filters."})
    tips.append({"type":"info","title":"Use strong action verbs",
        "text":"Start every bullet with: Built, Designed, Led, Reduced, Improved, Deployed, Automated."})
    tips.append({"type":"info","title":"Keep it concise",
        "text":"Recruiters spend ~7 seconds on a first scan. One to two pages maximum."})
    return tips

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded."}), 400
    file      = request.files["resume"]
    job_desc  = request.form.get("job_desc", "").strip()
    job_title = request.form.get("job_title", "Job Role").strip()

    if not file or file.filename == "":
        return jsonify({"error": "Please select a file."}), 400
    if not allowed(file.filename):
        return jsonify({"error": "Only PDF and DOCX files are accepted."}), 400
    if len(job_desc) < 30:
        return jsonify({"error": "Please enter a longer job description."}), 400

    fname = secure_filename(file.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(fpath)

    resume_text   = extract_text(fpath) or f"Resume: {fname}"
    resume_skills = extract_skills(resume_text)
    job_skills    = extract_skills(job_desc)
    matched = sorted(set(s for s in resume_skills if s in job_skills))
    missing = sorted(set(s for s in job_skills    if s not in resume_skills))
    score   = similarity_score(resume_text, job_desc)
    tips    = make_suggestions(score, matched, missing)
    edu     = extract_education(resume_text)
    exp     = extract_experience(resume_text)

    rec = {
        "id": str(uuid.uuid4()), "filename": fname, "job_title": job_title,
        "score": score, "matched": matched, "missing": missing,
        "suggestions": tips, "keywords": len(matched),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    db_save(rec)

    return jsonify({"id": rec["id"], "score": score, "matched": matched,
                    "missing": missing, "suggestions": tips,
                    "keywords": len(matched), "education": edu, "experience": exp})

@app.route("/history")
def history():
    return render_template("history.html", records=db_all())

@app.route("/result/<aid>")
def result(aid):
    rec = db_get(aid)
    if not rec: return redirect(url_for("index"))
    return render_template("result.html", rec=rec)

@app.route("/delete/<aid>", methods=["POST"])
def delete(aid):
    db_delete(aid); return redirect(url_for("history"))

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    init_db()
    app.run(debug=True)
