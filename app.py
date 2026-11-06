import os
import re
import json
import time
import base64
import random
import secrets
import threading
from functools import wraps
from urllib.parse import quote

import indexer
import retrieval
import requests
from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from embedder import Embedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import chromadb
from google import genai
from google.genai.errors import ServerError, ClientError

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap

client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = Embedder()

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="campus_docs")

# ---- warm-up: do the heavy first-time work at boot, not during the first question ----
print("[warmup] starting", flush=True)
try:
    _t = time.time()
    _warm_vec = model.encode("warm up").tolist()
    print(f"[warmup] embedding model ready ({time.time() - _t:.1f}s)", flush=True)
    _t = time.time()
    retrieval._load_corpus(collection)
    print(f"[warmup] corpus + keyword index ready ({time.time() - _t:.1f}s)", flush=True)
    _t = time.time()
    print("[warmup] loading vector index...", flush=True)
    collection.query(query_embeddings=[_warm_vec], n_results=1)
    print(f"[warmup] vector index ready ({time.time() - _t:.1f}s)", flush=True)
except Exception as _e:
    print(f"[warmup] failed: {_e}", flush=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ADMIN_FILE = os.path.join(BASE_DIR, "admins.json")
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
serializer = URLSafeTimedSerializer(SECRET_KEY)
TOKEN_MAX_AGE = 60 * 60 * 12  # 12 hours

BRANDING_FILE = os.path.join(BASE_DIR, "branding.json")
DEFAULT_BRANDING = {"name": "NEXUS", "tagline": ""}
MAX_NAME_LEN = 40
MAX_TAGLINE_LEN = 100


def bootstrap_branding():
    if os.path.exists(BRANDING_FILE):
        return
    with open(BRANDING_FILE, "w") as f:
        json.dump(DEFAULT_BRANDING, f, indent=2)


def load_branding():
    try:
        with open(BRANDING_FILE) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = dict(DEFAULT_BRANDING)
    return {
        "name": data.get("name") or DEFAULT_BRANDING["name"],
        "tagline": data.get("tagline", DEFAULT_BRANDING["tagline"])
    }


def save_branding(name, tagline):
    data = {"name": name, "tagline": tagline}
    with open(BRANDING_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return data


bootstrap_branding()


def bootstrap_admin():
    if os.path.exists(ADMIN_FILE):
        return

    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD")

    if not password:
        password = secrets.token_urlsafe(9)
        print("\n" + "=" * 60)
        print("No ADMIN_PASSWORD set in .env — generated one for first run:")
        print(f"  Username: {username}")
        print(f"  Password: {password}")
        print("Set ADMIN_USERNAME / ADMIN_PASSWORD in .env to keep this fixed.")
        print("=" * 60 + "\n")

    admins = {
        username: {
            "password_hash": generate_password_hash(password, method="pbkdf2:sha256"),
            "role": "admin"
        }
    }
    with open(ADMIN_FILE, "w") as f:
        json.dump(admins, f, indent=2)


def load_admins():
    with open(ADMIN_FILE) as f:
        return json.load(f)


bootstrap_admin()


def create_token(username, role):
    return serializer.dumps({"username": username, "role": role})


def verify_token(token):
    try:
        return serializer.loads(token, max_age=TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def require_admin(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header.split(" ", 1)[1]
        payload = verify_token(token)
        if not payload or payload.get("role") != "admin":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


USERS_FILE = os.path.join(BASE_DIR, "users.json")
ALLOW_SIGNUP = os.getenv("ALLOW_SIGNUP", "true").strip().lower() not in ("false", "0", "no")
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 128
_users_lock = threading.Lock()

_DUMMY_HASH = generate_password_hash("nexus-dummy-password", method="pbkdf2:sha256")

LOGIN_MAX_FAILURES = 8
LOGIN_WINDOW = 5 * 60
SIGNUP_MAX_PER_IP = 10
SIGNUP_WINDOW = 60 * 60
_attempts = {}
_attempts_lock = threading.Lock()


def _recent_attempts(key, window):
    now = time.time()
    with _attempts_lock:
        recent = [t for t in _attempts.get(key, []) if now - t < window]
        if recent:
            _attempts[key] = recent
        else:
            _attempts.pop(key, None)
        return len(recent)


def _record_attempt(key):
    with _attempts_lock:
        _attempts.setdefault(key, []).append(time.time())


def _clear_attempts(key):
    with _attempts_lock:
        _attempts.pop(key, None)


def load_users():
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_users(users):
    tmp_path = USERS_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp_path, USERS_FILE)


def find_admin(username):
    admins = load_admins()
    if username in admins:
        return username, admins[username]
    lowered = username.lower()
    for name, account in admins.items():
        if name.lower() == lowered:
            return name, account
    return None, None


def public_user(username, role, record):
    record = record or {}
    return {
        "username": username,
        "role": role,
        "name": record.get("name") or username,
        "enrol": record.get("enrol", ""),
        "course": record.get("course", ""),
    }


def require_login(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        payload = verify_token(auth_header.split(" ", 1)[1]) if auth_header.startswith("Bearer ") else None
        if not payload:
            return jsonify({"error": "Please log in to continue."}), 401
        if payload.get("role") != "admin" and payload.get("username") not in load_users():
            return jsonify({"error": "Please log in to continue."}), 401
        g.current_user = payload
        return f(*args, **kwargs)
    return wrapper


AI_PER_MINUTE = int(os.getenv("AI_PER_MINUTE", "15"))
AI_PER_DAY = int(os.getenv("AI_PER_DAY", "200"))


def ai_rate_limit_message():
    user = g.current_user["username"]
    minute_key, day_key = f"ai-min|{user}", f"ai-day|{user}"
    if _recent_attempts(minute_key, 60) >= AI_PER_MINUTE:
        return "You're asking too quickly. Please wait a minute and try again."
    if _recent_attempts(day_key, 24 * 60 * 60) >= AI_PER_DAY:
        return "You've reached today's question limit. Please try again tomorrow."
    _record_attempt(minute_key)
    _record_attempt(day_key)
    return None


POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"

ASPECT_RATIO_DIMENSIONS = {
    "1:1":  (1024, 1024),
    "2:3":  (768, 1152),
    "3:2":  (1152, 768),
    "3:4":  (896, 1195),
    "4:3":  (1195, 896),
    "4:5":  (896, 1120),
    "5:4":  (1120, 896),
    "9:16": (768, 1365),
    "16:9": (1365, 768),
    "21:9": (1344, 576),
}
ALLOWED_ASPECT_RATIOS = set(ASPECT_RATIO_DIMENSIONS.keys())


@app.route("/generate-image", methods=["POST"])
@require_login
def generate_image():
    data = request.get_json(silent=True) or {}

    prompt = (data.get("prompt") or "").strip()
    style = (data.get("style") or "").strip()
    aspect_ratio = (data.get("aspect_ratio") or "1:1").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    if len(prompt) > 4000:
        return jsonify({"error": "Prompt is too long. Maximum 4000 characters."}), 400

    limit_message = ai_rate_limit_message()
    if limit_message:
        return jsonify({"error": limit_message}), 429

    if aspect_ratio not in ALLOWED_ASPECT_RATIOS:
        aspect_ratio = "1:1"

    full_prompt = prompt
    if style:
        full_prompt += f", {style} style"

    width, height = ASPECT_RATIO_DIMENSIONS[aspect_ratio]
    seed = random.randint(0, 999_999)

    url = (
        f"{POLLINATIONS_BASE_URL}/{quote(full_prompt)}"
        f"?width={width}&height={height}&seed={seed}&nologo=true"
    )

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "image/jpeg")
        if not content_type.startswith("image/"):
            return jsonify({
                "error": "The image service returned something unexpected instead of an image."
            }), 502

        encoded = base64.b64encode(response.content).decode("ascii")
        return jsonify({
            "image": f"data:{content_type};base64,{encoded}",
            "model": "pollinations",
            "aspect_ratio": aspect_ratio,
        })

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "The image service took too long to respond. Please try again."
        }), 503

    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Could not reach the free image generation service. Please try again shortly.",
            "details": str(e),
        }), 503

    except Exception as e:
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Enter your username and password."}), 400

    throttle_key = f"login|{request.remote_addr}|{username.lower()}"
    if _recent_attempts(throttle_key, LOGIN_WINDOW) >= LOGIN_MAX_FAILURES:
        return jsonify({"error": "Too many failed attempts. Please wait a few minutes and try again."}), 429

    admin_name, admin_account = find_admin(username)
    if admin_account:
        canonical, role, record = admin_name, admin_account.get("role", "admin"), admin_account
        password_ok = check_password_hash(admin_account["password_hash"], password)
    else:
        canonical, role = username.lower(), "student"
        record = load_users().get(canonical)
        password_ok = check_password_hash(record["password_hash"] if record else _DUMMY_HASH, password) and record is not None

    if not password_ok:
        _record_attempt(throttle_key)
        return jsonify({"error": "Invalid username or password."}), 401

    _clear_attempts(throttle_key)
    token = create_token(canonical, role)
    return jsonify({"token": token, **public_user(canonical, role, record)})


@app.route("/signup", methods=["POST"])
def signup():
    if not ALLOW_SIGNUP:
        return jsonify({"error": "Sign-ups are turned off. Please ask an admin for an account."}), 403

    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    name = (data.get("name") or "").strip()[:60]
    enrol = (data.get("enrol") or "").strip()[:30]
    course = (data.get("course") or "").strip()[:60]

    if not name:
        return jsonify({"error": "Please enter your name."}), 400
    if not USERNAME_RE.match(username):
        return jsonify({"error": "Username must be 3-30 characters: letters, numbers, dots, dashes or underscores."}), 400
    if len(password) < MIN_PASSWORD_LEN:
        return jsonify({"error": f"Password must be at least {MIN_PASSWORD_LEN} characters."}), 400
    if len(password) > MAX_PASSWORD_LEN:
        return jsonify({"error": f"Password must be {MAX_PASSWORD_LEN} characters or fewer."}), 400

    throttle_key = f"signup|{request.remote_addr}"
    if _recent_attempts(throttle_key, SIGNUP_WINDOW) >= SIGNUP_MAX_PER_IP:
        return jsonify({"error": "Too many accounts created from this device. Please try again later."}), 429

    if find_admin(username)[1]:
        return jsonify({"error": "That username is already taken."}), 409

    with _users_lock:
        users = load_users()
        if username in users:
            return jsonify({"error": "That username is already taken."}), 409
        record = {
            "password_hash": generate_password_hash(password, method="pbkdf2:sha256"),
            "name": name,
            "enrol": enrol,
            "course": course,
            "created": int(time.time()),
        }
        users[username] = record
        save_users(users)

    _record_attempt(throttle_key)
    token = create_token(username, "student")
    return jsonify({"token": token, **public_user(username, "student", record)}), 201


@app.route("/me", methods=["GET"])
@require_login
def me():
    payload = g.current_user
    if payload.get("role") == "admin":
        name, record = find_admin(payload["username"])
        return jsonify(public_user(name or payload["username"], "admin", record))
    return jsonify(public_user(payload["username"], "student", load_users().get(payload["username"])))


@app.route("/admin/verify", methods=["GET"])
@require_admin
def verify_session():
    return jsonify({"valid": True})


@app.route("/admin/upload", methods=["POST"])
@require_admin
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(BASE_DIR, filename)
    file.save(save_path)

    try:
        count, courses_found = indexer.index_pdf_into_collection(
            save_path, filename, collection, model, splitter
        )
    except Exception as e:
        return jsonify({"error": f"Failed to index PDF: {str(e)}"}), 500

    _semester_code_cache.clear()

    return jsonify({
        "message": f"{filename} indexed successfully.",
        "chunks_added": count,
        "course_codes_found": sorted(courses_found)
    })


@app.route("/admin/documents", methods=["GET"])
@require_admin
def list_documents():
    all_data = collection.get(include=["metadatas"])
    counts = {}
    for meta in all_data["metadatas"]:
        src = meta.get("source", "Unknown")
        counts[src] = counts.get(src, 0) + 1
    docs = [{"source": src, "chunks": n} for src, n in sorted(counts.items())]
    return jsonify({"documents": docs})


@app.route("/config", methods=["GET"])
def get_config():
    config = load_branding()
    config["signup_enabled"] = ALLOW_SIGNUP
    return jsonify(config)


@app.route("/admin/branding", methods=["POST"])
@require_admin
def update_branding():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    tagline = (data.get("tagline") or "").strip()

    if not name:
        return jsonify({"error": "Name cannot be empty."}), 400
    if len(name) > MAX_NAME_LEN:
        return jsonify({"error": f"Name must be {MAX_NAME_LEN} characters or fewer."}), 400
    if len(tagline) > MAX_TAGLINE_LEN:
        return jsonify({"error": f"Tagline must be {MAX_TAGLINE_LEN} characters or fewer."}), 400

    updated = save_branding(name, tagline)
    return jsonify(updated)


@app.route("/admin/documents/<path:source>", methods=["DELETE"])
@require_admin
def delete_document(source):
    existing = collection.get(where={"source": source}, include=[])
    if not existing["ids"]:
        return jsonify({"error": f"No document found with source '{source}'"}), 404

    collection.delete(where={"source": source})
    _semester_code_cache.clear()

    return jsonify({
        "message": f"{source} deleted successfully.",
        "chunks_removed": len(existing["ids"])
    })


COURSE_CODE_QUERY_RE = re.compile(r"\bBCA\s?(\d{3})\s?([TP]?)\b", re.IGNORECASE)

SUBJECT_KEYWORDS = {
    "programming for problem solving using c": ["BCA101T", "BCA101P"],
    "problem solving using c": ["BCA101T", "BCA101P"],
    "c programming": ["BCA101T", "BCA101P"],
    "c language": ["BCA101T", "BCA101P"],
    "c": ["BCA101T", "BCA101P"],
    "programming in c": ["BCA101T", "BCA101P"],
    "information technology": ["BCA103T", "BCA103P"],
    "web technologies": ["BCA105T", "BCA105P"],
    "web development": ["BCA105T", "BCA105P"],
    "mathematical foundation": ["BCA107T"],
    "discrete mathematics": ["BCA107T"],
    "writing skills": ["BCA141T"],
    "bridge course in mathematics": ["BCA181T"],
    "database management": ["BCA102T", "BCA102P"],
    "database management system": ["BCA102T", "BCA102P"],
    "dbms": ["BCA102T", "BCA102P"],
    "object oriented programming": ["BCA104T", "BCA104P"],
    "object-oriented programming": ["BCA104T", "BCA104P"],
    "java": ["BCA104T", "BCA104P", "BCA309T", "BCA309P"],
    "oops": ["BCA104T", "BCA104P"],
    "data structures": ["BCA106T", "BCA106P"],
    "algorithms": ["BCA106T", "BCA106P"],
    "data structures and algorithms": ["BCA106T", "BCA106P"],
    "dsa": ["BCA106T", "BCA106P"],
    "software engineering": ["BCA108T", "BCA108P"],
    "soft skills": ["BCA142T"],
    "environment studies": ["BCA192T"],
    "environmental studies": ["BCA192T"],
    "python programming": ["BCA201T", "BCA201P"],
    "python": ["BCA201T", "BCA201P"],
    "dynamic web designing": ["BCA203T", "BCA203P"],
    "dynamic web design": ["BCA203T", "BCA203P"],
    "computer organization": ["BCA205T", "BCA205P"],
    "computer organization and architecture": ["BCA205T", "BCA205P"],
    "coa": ["BCA205T", "BCA205P"],
    "operating systems": ["BCA202T", "BCA202P"],
    "operating system": ["BCA202T", "BCA202P"],
    "os": ["BCA202T", "BCA202P"],
    "software testing": ["BCA204T", "BCA204P"],
    "computer networks": ["BCA301T", "BCA301P"],
    "networking": ["BCA301T", "BCA301P"],
    "artificial intelligence": ["BCA303T", "BCA303P"],
    "artificial intelligence ai": ["BCA303T", "BCA303P"],
    "ai": ["BCA303T", "BCA303P"],
    "natural language processing": ["BCA305T", "BCA305P"],
    "nlp": ["BCA305T", "BCA305P"],
    "network security": ["BCA307T", "BCA307P"],
    "full stack development": ["BCA309T", "BCA309P"],
    "full stack": ["BCA309T", "BCA309P"],
    "management and entrepreneurship": ["BCA313T"],
    "entrepreneurship": ["BCA313T"],
}


def find_course_codes_from_keywords(question):
    q = question.lower()
    codes = set()
    for phrase, phrase_codes in SUBJECT_KEYWORDS.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", q):
            codes.update(phrase_codes)
    return codes


def find_semester_keywords(question):
    q = question.lower()
    semester_map = {
        1: ["first semester", "1st semester", "semester 1", "sem 1", "sem1"],
        2: ["second semester", "2nd semester", "semester 2", "sem 2", "sem2"],
        3: ["third semester", "3rd semester", "semester 3", "sem 3", "sem3"],
        4: ["fourth semester", "4th semester", "semester 4", "sem 4", "sem4"],
        5: ["fifth semester", "5th semester", "semester 5", "sem 5", "sem5"],
        6: ["sixth semester", "6th semester", "semester 6", "sem 6", "sem6"],
        7: ["seventh semester", "7th semester", "semester 7", "sem 7", "sem7"],
        8: ["eighth semester", "8th semester", "semester 8", "sem 8", "sem8"],
    }
    for semester, keywords in semester_map.items():
        for keyword in keywords:
            if keyword in q:
                return semester
    return None


_semester_code_cache = {}
CODE_PATTERN = re.compile(r"^BCA(\d{3})[TP]?$")


def get_all_course_codes():
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    codes = set()
    for m in all_meta:
        code = m.get("course_code")
        if code:
            codes.add(code)
    return codes


def compute_semester_course_codes(semester):
    cache_key = semester
    if cache_key in _semester_code_cache:
        return _semester_code_cache[cache_key]

    hundreds = (semester + 1) // 2
    parity = semester % 2

    matched = []
    for code in get_all_course_codes():
        m = CODE_PATTERN.match(code)
        if not m:
            continue
        num = int(m.group(1))
        if num // 100 == hundreds and (num % 100) % 2 == parity:
            matched.append(code)

    _semester_code_cache[cache_key] = matched
    return matched

@app.route("/app", methods=["GET"])
def serve_frontend():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


TEXT_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def generate_with_retry(prompt, attempts_per_model=2, delay=2):
    last_error = None
    for model_name in TEXT_MODELS:
        for attempt in range(attempts_per_model):
            try:
                return client_genai.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
            except ServerError as e:
                last_error = e
                print(f"[Gemini 5xx] {model_name} attempt {attempt + 1}/{attempts_per_model}: {e}")
                time.sleep(delay)
            except ClientError as e:
                code = getattr(e, "code", None)
                if code in (429, 404):
                    last_error = e
                    reason = "out of free quota" if code == 429 else "not available"
                    print(f"[Gemini {code}] {model_name} {reason}, trying next model")
                    break
                raise
    raise last_error


def ai_error_answer(error):
    code = getattr(error, "code", None)
    if isinstance(error, ClientError) and code == 429:
        return "The free AI quota is used up for now. Please try again in a little while."
    if isinstance(error, ServerError):
        return "The AI model is temporarily overloaded on Google's end. Please try again in a few seconds."
    print(f"[Gemini error] {type(error).__name__} code={code}: {error}")
    return "The AI service couldn't process that request. Please let an admin know if this keeps happening."


MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 700
REWRITE_MODEL = TEXT_MODELS[-1]

FOLLOWUP_OPENERS = re.compile(
    r"^\s*(and|also|but|so|then|ok(ay)?|what about|how about|why|how so|tell me more|more|elaborate|"
    r"explain (it|that|this|more|further)|can you (explain|elaborate|simplify)|give (me )?(an )?example)\b",
    re.IGNORECASE,
)
FOLLOWUP_WORDS = re.compile(
    r"\b(it|its|that|this|those|these|them|they|same|above|previous|earlier|also|too|again|else|further|another|next|last)\b",
    re.IGNORECASE,
)

REWRITE_PROMPT = """You rewrite a student's latest message so it can be understood WITHOUT the earlier conversation.

Rules:
- Use the conversation only to resolve references such as "it", "that subject", "Unit 2", "what about sem 3".
- Keep subject names, semester numbers and unit numbers explicit in the rewritten question.
- If the latest message is already clear on its own, return it unchanged.
- Do not answer the question. Do not add anything the student did not ask for.
- Output ONLY the rewritten question, on a single line.

Conversation so far:
{history}

Latest message: {question}

Standalone question:"""


def clean_history(raw):
    if not isinstance(raw, list):
        return []
    cleaned = []
    for item in raw[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            cleaned.append({"role": role, "content": content[:MAX_HISTORY_CHARS]})
    return cleaned


def format_history(history):
    return "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in history
    )


def looks_like_followup(question):
    q = question.strip()
    if FOLLOWUP_OPENERS.search(q) or FOLLOWUP_WORDS.search(q):
        return True
    has_subject = bool(
        COURSE_CODE_QUERY_RE.search(q) or find_course_codes_from_keywords(q) or find_semester_keywords(q)
    )
    return len(q.split()) <= 8 and not has_subject


def rewrite_followup(question, history):
    if not history or not looks_like_followup(question):
        return question
    prompt = REWRITE_PROMPT.format(history=format_history(history), question=question)
    try:
        response = client_genai.models.generate_content(model=REWRITE_MODEL, contents=prompt)
        lines = [ln.strip() for ln in (response.text or "").splitlines() if ln.strip()]
        rewritten = lines[0] if lines else ""
    except Exception as e:
        print(f"[rewrite] failed, using the original question: {e}")
        return question
    rewritten = re.sub(r"^(standalone|rewritten)?\s*question\s*:\s*", "", rewritten, flags=re.IGNORECASE)
    rewritten = rewritten.strip().strip("\"'").strip()
    if not rewritten or len(rewritten) > 400:
        return question
    return rewritten


STUDY_ACTION_INSTRUCTIONS = {
    "ask": "Answer the student's question directly and clearly using the academic context.",
    "explain": "Explain the topic in a simple, beginner-friendly way. Use short paragraphs and include at least one concrete example.",
    "summarize": "Produce concise, well-organized revision notes on the topic using Markdown headings and bullet points. Keep it scannable, not long paragraphs.",
    "quiz": "Create a short practice quiz (around 5 questions) on the topic. Put the correct answer and a one-line explanation directly under each question.",
    "flashcards": "Create around 8 flashcards on the topic. Format each as '**Q:** ...' on one line followed by '**A:** ...' on the next, covering key definitions and concepts.",
    "important": "Identify and list the most important topics or subtopics a student should prioritize revising, based only on what's in the academic context. Briefly say why each one matters.",
}

WRITTEN_SEMESTERS = {
    1: "FIRST SEMESTER", 2: "SECOND SEMESTER", 3: "THIRD SEMESTER", 4: "FOURTH SEMESTER",
    5: "FIFTH SEMESTER", 6: "SIXTH SEMESTER", 7: "SEVENTH SEMESTER", 8: "EIGHTH SEMESTER",
}


@app.route("/study-session", methods=["POST"])
@require_login
def study_session():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    action = data.get("action") or "ask"
    semester_raw = data.get("semester")

    if not question:
        return jsonify({"error": "Please enter something to study."}), 400

    limit_message = ai_rate_limit_message()
    if limit_message:
        return jsonify({"answer": limit_message, "sources": []}), 200

    if action not in STUDY_ACTION_INSTRUCTIONS:
        action = "ask"

    try:
        semester = int(semester_raw) if semester_raw not in (None, "", "null") else None
    except (TypeError, ValueError):
        semester = None

    question_embedding = model.encode(question).tolist()

    hit_docs, hit_metas = retrieval.hybrid_search(collection, question, question_embedding)
    all_docs = list(hit_docs)
    all_metas = list(hit_metas)

    def add_unique(docs, metas):
        for doc, meta in zip(docs, metas):
            if doc not in all_docs:
                all_docs.append(doc)
                all_metas.append(meta)

    if semester and semester in WRITTEN_SEMESTERS:
        semester_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=6,
            where_document={"$contains": WRITTEN_SEMESTERS[semester]}
        )
        add_unique(semester_results["documents"][0], semester_results["metadatas"][0])

        sem_codes = compute_semester_course_codes(semester)
        if sem_codes:
            sem_code_results = collection.get(where={"course_code": {"$in": sem_codes}})
            add_unique(sem_code_results["documents"], sem_code_results["metadatas"])

    candidates = find_course_codes_from_keywords(question)
    course_code_match = COURSE_CODE_QUERY_RE.search(question)
    if course_code_match:
        num, suffix = course_code_match.group(1), course_code_match.group(2).upper()
        if suffix:
            candidates.add(f"BCA{num}{suffix}")
        else:
            candidates.update([f"BCA{num}T", f"BCA{num}P"])

    if candidates:
        code_results = collection.get(where={"course_code": {"$in": list(candidates)}})
        add_unique(code_results["documents"], code_results["metadatas"])

    MAX_CONTEXT_CHUNKS = 80 if semester else 18
    if len(all_docs) > MAX_CONTEXT_CHUNKS:
        all_docs = all_docs[:MAX_CONTEXT_CHUNKS]
        all_metas = all_metas[:MAX_CONTEXT_CHUNKS]

    context_parts = []
    for doc, meta in zip(all_docs, all_metas):
        context_parts.append(f"{retrieval.chunk_header(meta)}\n{doc}")
    context = "\n\n".join(context_parts)

    sources = list(dict.fromkeys(meta.get("source", "Unknown source") for meta in all_metas))
    citations = retrieval.build_citations(all_metas)

    ai_name = load_branding()["name"]
    instruction = STUDY_ACTION_INSTRUCTIONS[action]

    prompt = f"""
You are {ai_name}, an academic study assistant for BCA students.

Task: {instruction}

Use ONLY the academic context provided below — do not invent information that isn't in it.
If the context doesn't cover the topic, clearly say so instead of guessing.
Format the response with Markdown (headings/bullets) where it helps readability.

Academic Context:
{context}

Student's topic / question:
{question}

Response:
"""

    try:
        response = generate_with_retry(prompt)
        answer_text = response.text or "The model returned an empty response — try narrowing the topic."
        return jsonify({"answer": answer_text, "sources": sources, "citations": citations})
    except (ServerError, ClientError) as e:
        return jsonify({"answer": ai_error_answer(e), "sources": sources}), 200


@app.route("/ask", methods=["POST"])
@require_login
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question.", "sources": []}), 400

    limit_message = ai_rate_limit_message()
    if limit_message:
        return jsonify({"answer": limit_message, "sources": [], "is_error": True}), 200

    _t0 = time.time()
    print(f"[ask] 1/5 received: {question[:60]!r}", flush=True)

    history = clean_history(data.get("history"))
    use_history = bool(history) and looks_like_followup(question)
    standalone = rewrite_followup(question, history)
    print(f"[ask] 2/5 question rewritten ({time.time() - _t0:.1f}s)", flush=True)

    question_embedding = model.encode(standalone).tolist()
    print(f"[ask] 3/5 embedded ({time.time() - _t0:.1f}s)", flush=True)

    hit_docs, hit_metas = retrieval.hybrid_search(collection, standalone, question_embedding)
    print(f"[ask] 4/5 retrieval done, {len(hit_docs)} chunks ({time.time() - _t0:.1f}s)", flush=True)
    all_docs = list(hit_docs)
    all_metas = list(hit_metas)

    def add_unique(docs, metas):
        for doc, meta in zip(docs, metas):
            if doc not in all_docs:
                all_docs.append(doc)
                all_metas.append(meta)

    semester = find_semester_keywords(standalone)
    if semester:
        written_semesters = {
            1: "FIRST SEMESTER", 2: "SECOND SEMESTER", 3: "THIRD SEMESTER", 4: "FOURTH SEMESTER",
            5: "FIFTH SEMESTER", 6: "SIXTH SEMESTER", 7: "SEVENTH SEMESTER", 8: "EIGHTH SEMESTER"
        }
        semester_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=6,
            where_document={"$contains": written_semesters[semester]}
        )
        add_unique(semester_results["documents"][0], semester_results["metadatas"][0])

        sem_codes = compute_semester_course_codes(semester)
        if sem_codes:
            sem_code_results = collection.get(where={"course_code": {"$in": sem_codes}})
            add_unique(sem_code_results["documents"], sem_code_results["metadatas"])

    candidates = set()
    course_code_match = COURSE_CODE_QUERY_RE.search(standalone)
    if course_code_match:
        num, suffix = course_code_match.group(1), course_code_match.group(2).upper()
        if suffix:
            candidates.add(f"BCA{num}{suffix}")
        else:
            candidates.update([f"BCA{num}T", f"BCA{num}P"])
    candidates.update(find_course_codes_from_keywords(standalone))

    if candidates:
        code_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=6,
            where={"course_code": {"$in": list(candidates)}}
        )
        add_unique(code_results["documents"][0], code_results["metadatas"][0])

    MAX_CONTEXT_CHUNKS = 80 if semester else 18
    if len(all_docs) > MAX_CONTEXT_CHUNKS:
        all_docs = all_docs[:MAX_CONTEXT_CHUNKS]
        all_metas = all_metas[:MAX_CONTEXT_CHUNKS]

    context_parts = []
    for doc, meta in zip(all_docs, all_metas):
        context_parts.append(f"{retrieval.chunk_header(meta)}\n{doc}")
    context = "\n\n".join(context_parts)

    sources = list(dict.fromkeys(meta.get("source", "Unknown source") for meta in all_metas))
    citations = retrieval.build_citations(all_metas)

    ai_name = load_branding()["name"]

    history_block = (
        "Earlier in this conversation (for understanding only):\n" + format_history(history) + "\n\n"
        if use_history else ""
    )

    prompt = f"""
You are {ai_name}, an academic assistant for BCA students.

Answer the student's question using ONLY the academic context provided below.

Rules:
- Do not invent information.
- If the answer is not present in the context, clearly say that the information could not be found in the available academic documents.
- Prefer information that directly matches the student's semester, subject, course code, or topic.
- If multiple relevant pieces of information are available, combine them clearly.
- When discussing syllabus information, preserve the Unit structure where possible.
- When discussing credits, clearly identify the subject and credit value.
- Cite your sources inline in square brackets using the file name and page from the context headers, e.g. [BCA_1st_year.pdf, p. 3] or [BCASYL.pdf, pp. 11-12]. Only cite pages that appear in the context headers; never guess a page number. If a context block has no page, cite just the file name.
- Keep the answer clear, organized, and easy for a BCA student to understand.
- Earlier messages are only there to help you understand the question. Never treat earlier assistant replies as facts: check everything against the Academic Context.

Academic Context:
{context}

{history_block}Student Question:
{standalone}

Answer:
"""

    print(f"[ask] 5/5 calling Gemini, prompt {len(prompt)} chars ({time.time() - _t0:.1f}s)", flush=True)
    try:
        response = generate_with_retry(prompt)
        print(f"[ask] done: Gemini replied ({time.time() - _t0:.1f}s)", flush=True)
        answer_text = response.text or "The model returned an empty response — please try rephrasing your question or asking about a narrower topic (e.g. one subject instead of a whole semester)."
        payload = {"answer": answer_text, "sources": sources, "citations": citations}
        if standalone != question:
            payload["standalone_question"] = standalone
        return jsonify(payload)
    except (ServerError, ClientError) as e:
        return jsonify({"answer": ai_error_answer(e), "sources": sources, "is_error": True}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=os.getenv("FLASK_DEBUG", "0") == "1")