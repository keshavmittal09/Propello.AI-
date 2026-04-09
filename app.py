import os
import re
import json
import urllib.parse
import urllib.request
import urllib.error
import tempfile
import fitz  # PyMuPDF
import docx  # python-docx
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from openai import OpenAI

# ── Google Sheets (optional — only active when env vars are set) ───────────────
try:
    import gspread
    from google.oauth2.service_account import Credentials as GCredentials
    _GSPREAD_AVAILABLE = True
except ImportError:
    _GSPREAD_AVAILABLE = False





# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Propello AI", description="Real Estate Chatbot API", version="1.0.0")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
ROOT_PROJECTS_FILE = BASE_DIR / "projects.json"
DATA_PROJECTS_FILE = DATA_DIR / "projects.json"
ENV_FILE = BASE_DIR / ".env"


def load_local_env() -> None:
    """Load key=value pairs from .env into process env if not already set."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()

ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

# ── Google Sheets client (lazy, cached) ───────────────────────────────────────
_gs_client = None
_gs_spreadsheet = None

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

LEADS_SHEET_NAME    = "Leads"
PROJECTS_SHEET_NAME = "Projects"
LEADS_HEADERS       = ["Timestamp", "Last_Updated", "Name", "Phone", "Budget", "Location", "Appointment"]


def _use_sheets() -> bool:
    """Return True only when gspread is installed and credentials are provided."""
    return _GSPREAD_AVAILABLE and bool(
        os.environ.get("GOOGLE_CREDENTIALS_JSON") and
        os.environ.get("SPREADSHEET_ID")
    )


def _get_spreadsheet():
    """Return (and cache) the authenticated gspread Spreadsheet object."""
    global _gs_client, _gs_spreadsheet
    if _gs_spreadsheet is not None:
        return _gs_spreadsheet
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
    creds_dict = json.loads(creds_json)
    creds = GCredentials.from_service_account_info(creds_dict, scopes=SCOPES)
    _gs_client = gspread.authorize(creds)
    spreadsheet_id = os.environ.get("SPREADSHEET_ID")
    _gs_spreadsheet = _gs_client.open_by_key(spreadsheet_id)
    # Ensure both sheets exist
    existing = [ws.title for ws in _gs_spreadsheet.worksheets()]
    if LEADS_SHEET_NAME not in existing:
        ws = _gs_spreadsheet.add_worksheet(title=LEADS_SHEET_NAME, rows=1000, cols=20)
        ws.append_row(LEADS_HEADERS)
    if PROJECTS_SHEET_NAME not in existing:
        _gs_spreadsheet.add_worksheet(title=PROJECTS_SHEET_NAME, rows=500, cols=2)
    return _gs_spreadsheet


def _leads_sheet():
    return _get_spreadsheet().worksheet(LEADS_SHEET_NAME)


def _projects_sheet():
    return _get_spreadsheet().worksheet(PROJECTS_SHEET_NAME)

def resolve_projects_file() -> Path:
    if DATA_PROJECTS_FILE.exists():
        return DATA_PROJECTS_FILE
    if ROOT_PROJECTS_FILE.exists():
        return ROOT_PROJECTS_FILE
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_PROJECTS_FILE


def get_projects_files() -> list[Path]:
    files: list[Path] = []
    if DATA_PROJECTS_FILE.exists():
        files.append(DATA_PROJECTS_FILE)
    if ROOT_PROJECTS_FILE.exists() and ROOT_PROJECTS_FILE not in files:
        files.append(ROOT_PROJECTS_FILE)
    if not files:
        files.append(resolve_projects_file())
    return files


PROJECTS_FILE = resolve_projects_file()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# client        = anthropic.Anthropic(api_key=os.environ.get("groq_api"))


def get_llm_client() -> OpenAI:
    """Create an OpenAI-compatible client for Groq using available env vars."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing API key. Set GROQ_API_KEY in your environment.",
        )
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def get_chat_models() -> list[str]:
    """Return primary + fallback chat models (deduplicated, in order)."""
    primary = (os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
    fallback_raw = (os.environ.get("GROQ_FALLBACK_MODELS") or "llama-3.1-8b-instant,gemma2-9b-it").strip()

    models: list[str] = []
    if primary:
        models.append(primary)
    for part in fallback_raw.split(","):
        model = part.strip()
        if model:
            models.append(model)

    deduped: list[str] = []
    for model in models:
        if model not in deduped:
            deduped.append(model)
    return deduped


def create_chat_completion_with_fallback(
    client: OpenAI,
    messages: list[dict],
    max_tokens: int,
    stream: bool = False,
):
    """Try primary model first, then configured fallbacks on quota/model failures."""
    models = get_chat_models()
    rate_limited = False
    last_error = None

    for model in models:
        try:
            return client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                stream=stream,
            )
        except Exception as exc:
            last_error = exc
            text = str(exc).lower()
            if "rate limit" in text or "rate_limit" in text or "429" in text:
                rate_limited = True
                continue
            if "model" in text and ("not found" in text or "does not exist" in text or "unknown" in text):
                continue
            # Other API errors can still recover on fallback models.
            continue

    if rate_limited:
        raise RuntimeError("Priya is temporarily rate-limited on AI tokens. Please retry in a few minutes.")
    if last_error:
        raise RuntimeError(f"AI request failed: {last_error}")
    raise RuntimeError("AI request failed: no model could generate a response.")


def has_llm_key() -> bool:
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def has_elevenlabs_key() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY"))

LANGUAGES = {
    "en": {"label": "English",   "native": "English"},
    "hi": {"label": "Hindi",     "native": "हिन्दी"},
    "mr": {"label": "Marathi",   "native": "मराठी"},
    "gu": {"label": "Gujarati",  "native": "ગુજરાતી"},
    "ta": {"label": "Tamil",     "native": "தமிழ்"},
    "te": {"label": "Telugu",    "native": "తెలుగు"},
    "bn": {"label": "Bengali",   "native": "বাংলা"},
    "kn": {"label": "Kannada",   "native": "ಕನ್ನಡ"},
    "ml": {"label": "Malayalam", "native": "മലയാളം"},
    "pa": {"label": "Punjabi",   "native": "ਪੰਜਾਬੀ"},
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class LeadInfo(BaseModel):
    name:     Optional[str] = None
    phone:    Optional[str] = None
    budget:   Optional[str] = None
    location: Optional[str] = None
    appointment: Optional[str] = None

import csv
from datetime import datetime
LEADS_FILE = DATA_DIR / "leads.csv"

def get_booked_appointments() -> list[str]:
    """Return a list of all booked appointment strings (from Sheets or CSV)."""
    if _use_sheets():
        try:
            ws = _leads_sheet()
            records = ws.get_all_records()
            return [r["Appointment"] for r in records if r.get("Appointment")]
        except Exception as e:
            print(f"Sheets read error (appointments): {e}")
            return []
    # Fallback: local CSV
    if not LEADS_FILE.exists():
        return []
    booked = []
    try:
        with open(LEADS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                apt = row.get("Appointment")
                if apt and apt.strip():
                    booked.append(apt.strip())
    except Exception:
        pass
    return booked


def log_lead_to_csv(lead: LeadInfo):
    """Save / update lead in Google Sheets (if configured) or local CSV."""
    if not any([lead.name, lead.phone, lead.budget, lead.location, lead.appointment]):
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if _use_sheets():
        try:
            ws = _leads_sheet()
            records = ws.get_all_records()
            headers = LEADS_HEADERS
            updated_row_idx = None
            for idx, row in enumerate(records):
                match_phone = (lead.phone and row.get("Phone") == lead.phone)
                match_name  = (lead.name and row.get("Name") == lead.name and not lead.phone)
                if match_phone or match_name:
                    updated_row_idx = idx + 2  # +1 header, +1 1-indexed
                    break
            if updated_row_idx:
                # Read existing row then merge
                existing = ws.row_values(updated_row_idx)
                # pad to header length
                while len(existing) < len(headers):
                    existing.append("")
                row_dict = dict(zip(headers, existing))
                if lead.name:        row_dict["Name"]         = lead.name
                if lead.phone:       row_dict["Phone"]        = lead.phone
                if lead.budget:      row_dict["Budget"]       = lead.budget
                if lead.location:    row_dict["Location"]     = lead.location
                if lead.appointment: row_dict["Appointment"]  = lead.appointment
                row_dict["Last_Updated"] = now
                ws.update(f"A{updated_row_idx}", [[row_dict.get(h, "") for h in headers]])
            else:
                ws.append_row([
                    now, now,
                    lead.name or "", lead.phone or "",
                    lead.budget or "", lead.location or "",
                    lead.appointment or ""
                ])
        except Exception as e:
            print(f"Sheets write error (lead): {e}")
        return

    # ── Fallback: local CSV ────────────────────────────────────────────────────
    LEADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if LEADS_FILE.exists():
        try:
            with open(LEADS_FILE, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            pass
    updated = False
    for row in rows:
        match_phone = (lead.phone and row.get("Phone") == lead.phone)
        match_name  = (lead.name and row.get("Name") == lead.name and not lead.phone)
        if match_phone or match_name:
            if lead.name:        row["Name"]        = lead.name
            if lead.phone:       row["Phone"]       = lead.phone
            if lead.budget:      row["Budget"]      = lead.budget
            if lead.location:    row["Location"]    = lead.location
            if lead.appointment: row["Appointment"] = lead.appointment
            row["Last_Updated"] = now
            updated = True
            break
    if not updated:
        rows.append({
            "Timestamp": now, "Last_Updated": now,
            "Name": lead.name or "", "Phone": lead.phone or "",
            "Budget": lead.budget or "", "Location": lead.location or "",
            "Appointment": lead.appointment or ""
        })
    try:
        with open(LEADS_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEADS_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        print(f"Warning: {LEADS_FILE} is open in another program. Please close it.")
    except Exception as e:
        print(f"Error saving lead: {e}")


def log_lead_to_crm(lead: LeadInfo, transcript: str = "") -> bool:
    """Best-effort sync of lead data to external CRM webhook."""
    url = (os.environ.get("CRM_WEBHOOK_URL") or "").strip()
    if not url:
        return False

    if not any([lead.name, lead.phone, lead.budget, lead.location, lead.appointment]):
        return False

    if not (lead.phone or "").strip():
        return False

    def _resolve_webhook_url(raw_url: str) -> str:
        parsed = urllib.parse.urlparse(raw_url)
        path = (parsed.path or "").rstrip("/").lower()
        if path in ("/api/priya/lead-captured", "/api/leads/inbound", "/api/webhooks/priya"):
            return raw_url
        return raw_url.rstrip("/") + "/api/priya/lead-captured"

    def _parse_budget_range(raw: Optional[str]) -> tuple[Optional[float], Optional[float]]:
        if not raw:
            return None, None
        text = str(raw).lower().replace(",", "").replace("₹", "")
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*(cr|crore|l|lac|lakh)?", text)
        if not matches:
            return None, None

        values = []
        for num, unit in matches:
            value = float(num)
            if unit in ("cr", "crore"):
                value *= 10_000_000
            elif unit in ("l", "lac", "lakh"):
                value *= 100_000
            values.append(value)

        if len(values) == 1:
            return None, values[0]
        return min(values[0], values[1]), max(values[0], values[1])

    budget_min, budget_max = _parse_budget_range(lead.budget)

    appointment_note = f"Appointment requested: {lead.appointment}" if lead.appointment else None
    notes = "\n".join(x for x in [appointment_note, transcript[:1200] if transcript else None] if x)

    payload = {
        "source": "priya_ai",
        "name": lead.name or "Unknown Lead",
        "phone": (lead.phone or "").strip(),
        "budget_min": budget_min,
        "budget_max": budget_max,
        "location_preference": lead.location,
        "timeline": lead.appointment,
        "transcript_summary": transcript[:1200] if transcript else None,
        "personal_notes": notes or None,
    }

    secret = (os.environ.get("CRM_WEBHOOK_SECRET") or os.environ.get("PRIYA_WEBHOOK_SECRET") or "").strip()
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["x-priya-secret"] = secret

    try:
        req = urllib.request.Request(
            _resolve_webhook_url(url),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"CRM webhook error: {e}")
        return False


def persist_lead(lead: LeadInfo, transcript: str = "") -> None:
    """Persist lead to local storage and external CRM (if configured)."""
    log_lead_to_csv(lead)
    log_lead_to_crm(lead, transcript)


class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    project_id: Optional[str] = None
    language:   str = "auto"
    history:    list[ChatMessage] = Field(default_factory=list)
    message:    str
    lead:       LeadInfo = Field(default_factory=LeadInfo)
    voice_mode: bool = False


class ConfigItem(BaseModel):
    type:  str
    size:  str = ""
    price: str = ""


class NearbyItem(BaseModel):
    name:     str
    distance: str = ""


class AddProjectRequest(BaseModel):
    name:         str
    developer:    str
    tagline:      str
    location:     str
    logo_emoji:   str = "🏢"
    accent_color: str = "#1a6b5a"
    configs:      list[ConfigItem] = Field(default_factory=list)
    highlights:   list[str] = Field(default_factory=list)
    amenities:    dict[str, list[str]] = Field(default_factory=dict)
    nearby:       list[NearbyItem] = Field(default_factory=list)
    connectivity: list[NearbyItem] = Field(default_factory=list)
    payment_plan: str = "Flexible plans available"
    contact:      str = ""


class VoiceTTSRequest(BaseModel):
    text: str
    language: str = "en"


def _build_elevenlabs_tts_request(text: str) -> urllib.request.Request:
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.82,
            "style": 0.25,
            "use_speaker_boost": True,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    api_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY", "")
    return urllib.request.Request(
        url,
        data=data,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_projects() -> list[dict]:
    """Load projects from Google Sheets (if configured) or local JSON files."""
    if _use_sheets():
        try:
            ws = _projects_sheet()
            rows = ws.get_all_values()  # [[id, json_str], ...]
            projects = []
            seen: set[str] = set()
            for row in rows:
                if len(row) < 2 or not row[0]:
                    continue
                pid = str(row[0]).strip()
                if pid in seen:
                    continue
                try:
                    project = json.loads(row[1])
                    seen.add(pid)
                    projects.append(project)
                except Exception:
                    pass
            return projects
        except Exception as e:
            print(f"Sheets read error (projects): {e}")
            return []

    # ── Fallback: local JSON files ────────────────────────────────────────
    merged: list[dict] = []
    seen_ids: set[str] = set()
    for projects_file in get_projects_files():
        if not projects_file.exists():
            continue
        try:
            with open(projects_file, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id", "")).strip()
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            merged.append(item)
    return merged


def save_projects(projects: list[dict]) -> None:
    """Save projects to Google Sheets (if configured) or local JSON file."""
    if _use_sheets():
        try:
            ws = _projects_sheet()
            ws.clear()
            rows = [[p.get("id", ""), json.dumps(p, ensure_ascii=False)] for p in projects]
            if rows:
                ws.update("A1", rows)
        except Exception as e:
            print(f"Sheets write error (projects): {e}")
        return

    # Fallback: local JSON
    projects_file = resolve_projects_file()
    projects_file.parent.mkdir(parents=True, exist_ok=True)
    with open(projects_file, "w", encoding="utf-8") as f:
        json.dump(projects, f, indent=2, ensure_ascii=False)


def build_project_brief(p: dict) -> str:
    configs = "\n".join(
        f"  - {c['type']}: {c.get('size','')}, priced at {c.get('price','')}"
        for c in p.get("configs", [])
    )
    amenities_block = ""
    for level, items in p.get("amenities", {}).items():
        amenities_block += f"\n  {level}: {', '.join(items)}"

    nearby = "\n".join(
        f"  - {n['name']}: {n.get('distance','')}" for n in p.get("nearby", [])
    )
    connectivity = "\n".join(
        f"  - {c['name']}: {c.get('distance','')}" for c in p.get("connectivity", [])
    )
    highlights = "\n".join(f"  - {h}" for h in p.get("highlights", []))

    return f"""
PROJECT: {p['name']} by {p['developer']}
Tagline: {p['tagline']}
Location: {p['location']}

CONFIGURATIONS & PRICING:
{configs}

KEY HIGHLIGHTS:
{highlights}

AMENITIES ({len(p.get('amenities', {}))} levels):
{amenities_block}

NEARBY LANDMARKS:
{nearby}

CONNECTIVITY:
{connectivity}

PAYMENT: {p.get('payment_plan', 'Flexible plans available')}
CONTACT: {p.get('contact', 'Contact sales team')}
""".strip()


def build_portfolio_brief(projects: list[dict]) -> str:
    return "\n\n".join(build_project_brief(project) for project in projects)


def detect_language_from_text(text: str, fallback: str = "en") -> str:
    """Detect a supported language from user text using script and keyword heuristics."""
    if not text:
        return fallback

    # Script-based detection first (high confidence).
    script_patterns = [
        (r"[\u0A80-\u0AFF]", "gu"),  # Gujarati
        (r"[\u0B80-\u0BFF]", "ta"),  # Tamil
        (r"[\u0C00-\u0C7F]", "te"),  # Telugu
        (r"[\u0980-\u09FF]", "bn"),  # Bengali
        (r"[\u0C80-\u0CFF]", "kn"),  # Kannada
        (r"[\u0D00-\u0D7F]", "ml"),  # Malayalam
        (r"[\u0A00-\u0A7F]", "pa"),  # Punjabi (Gurmukhi)
    ]
    for pattern, code in script_patterns:
        if re.search(pattern, text):
            return code

    if re.search(r"[\u0900-\u097F]", text):  # Devanagari -> Hindi/Marathi
        marathi_markers = (
            " मी ", " मला ", " आहे ", " तुम्ही ", "करा", "पाहिजे", "मराठी"
        )
        lowered = f" {text.lower()} "
        if any(marker in lowered for marker in marathi_markers):
            return "mr"
        return "hi"

    # Basic Romanized hints for common language switches.
    lowered = text.lower()
    roman_hints = {
        "hi": ["namaste", "kaise", "mujhe", "chahiye", "kya", "bhai"],
        "mr": ["namaskar", "mala", "pahije", "ahe", "marathi"],
        "ta": ["vanakkam", "enna", "venum"],
        "te": ["namaskaram", "kavali", "enti"],
        "bn": ["nomoskar", "amar", "chai"],
        "gu": ["kem", "majama", "mane", "joie"],
        "pa": ["sat sri akal", "menu", "chahida"],
    }
    for code, hints in roman_hints.items():
        if any(hint in lowered for hint in hints):
            return code

    return fallback


def resolve_chat_language(preferred: str, message: str, history: list[ChatMessage]) -> str:
    if preferred and preferred != "auto" and preferred in LANGUAGES:
        return preferred

    recent_user_messages = [m.content for m in history[-6:] if m.role == "user"]
    sample = "\n".join([*recent_user_messages, message])
    return detect_language_from_text(sample, fallback="en")


def build_system_prompt(
    projects: list[dict],
    language: str,
    lead: LeadInfo,
    voice_mode: bool = False,
) -> str:
    lang_name     = LANGUAGES.get(language, LANGUAGES["en"])["label"]
    project_brief = build_portfolio_brief(projects)
    developer_names = sorted({project.get("developer", "") for project in projects if project.get("developer")})
    developer_line = ", ".join(developer_names) if developer_names else "our developer partners"
    scope_line = "You are helping the user explore all available projects to find their extremely luxurious dream home."

    known = []
    if lead.name:
        known.append(f"You already know their name is {lead.name} — address them warmly.")
    if lead.phone:
        known.append("You already have their phone number.")
    if lead.budget:
        known.append(f"Their allocated budget is {lead.budget}.")
    if lead.location:
        known.append(f"They are interested in properties located in/around {lead.location}.")

    lead_context = "\n".join(known) if known else "You don't know anything about the user yet."

    missing = []
    if not lead.name:     missing.append("Name")
    if not lead.phone:    missing.append("Phone number")
    if not lead.budget:   missing.append("Budget constraint")
    if not lead.location: missing.append("Preferred Location or City")

    booked_apts = get_booked_appointments()
    apts_str = f"Currently booked appointments: {', '.join(booked_apts)}. NEVER double book these exact dates and times. Recommend an alternative time to the user if they request a clashing time." if booked_apts else "No existing appointments today."

    collect_str = (
        f"Missing crucial lead details: {', '.join(missing)}. Weave one of these naturally into the conversation. "
        "Never interrogate. Say things like, 'By the way, roughly what budget range should we look at?'\n"
        "STRICT RULE: If the user asks to book an appointment or site visit, you MUST tell them you cannot book it until they provide their Name, Phone, Budget, and Location. Do not confirm the booking if any of these are missing!"
        if missing else
        "You have all the lead info you need. Focus completely on closing — firmly invite them for a highly exclusive site visit and schedule an appointment."
    )

    voice_mode_rules = """
VOICE CONVERSATION MODE:
- This is a live voice conversation. Sound exactly like a completely authentic real human on a phone call.
- Express empathy and emotion natively.
- Use natural spoken language, NOT written style. Do not use any lists or bullet points.
- Keep each turn to 1-3 short spoken sentences.
- Add occasional human filler words naturally (e.g., "Ah, I see", "That's a great question", "Absolutely").
- Start with a brief acknowledgement, give the answer briefly, and ask one natural follow-up question.
""" if voice_mode else ""

    return f"""You are Priya, an incredibly empathetic, top-tier Real Estate Sales Manager at Propello AI representing {developer_line}.
{scope_line}

LANGUAGE: You MUST formulate your entire response strictly in fluent {lang_name}. CRITICAL: You MUST use the Latin/English alphabet ONLY (e.g., Romanized transliteration like "Namaste, kaise hain aap?"). NEVER use native scripts like Devanagari, Gujarati, or Tamil scripts, because the fallback audio engine cannot read them.

WHAT YOU KNOW ABOUT THE CLIENT:
{lead_context}

YOUR CORE SALES GOAL:
{collect_str}

Whenever you detect the user sharing any of these details, silently note them by appending these exact tags exactly ONCE at the VERY END of your absolutely final response paragraph, on a new line:
[LEAD:name=VALUE], [LEAD:phone=VALUE], [LEAD:budget=VALUE], [LEAD:location=VALUE], [LEAD:appointment=Date and Time]

{apts_str}

PROJECT PORTFOLIO FACT SHEET:
{project_brief}

YOUR PERSONA & CONVERSATIONAL STYLE:
- **EXTREMELY NOVICE-FRIENDLY & SIMPLE:** Use very simple, everyday words. Avoid complex real estate jargon. Talk like you are explaining to a friend.
- **VERY CONCISE:** Keep every response incredibly short—only 2 or 3 brief sentences max. Never overwhelm the user. Do not give too many details at once.
- **NO MARKDOWN EVER:** Never use bullet points, bolding, italics, or headers. Write in short, flowing text. 
- **ACTIVE LISTENING:** Affirm what they said smoothly.
- **B.A.C. (Be Always Closing):** Smoothly guide them toward giving their phone number or booking a site visit with ONE simple follow-up question.
{voice_mode_rules}
"""


def parse_lead_tags(raw: str, existing: LeadInfo) -> tuple[str, LeadInfo]:
    """Strip [LEAD:key=value] tags from raw text and merge into existing lead."""
    updated = existing.model_dump()
    for key in ("name", "phone", "budget", "location", "appointment"):
        match = re.search(rf"\[LEAD:{key}=([^\]]+)\]", raw)
        if match and not updated.get(key):
            updated[key] = match[1].strip()
    clean = re.sub(r"\[LEAD:[^\]]+\]", "", raw).strip()
    return clean, LeadInfo(**updated)


def extract_lead_from_user_message(message: str, existing: LeadInfo) -> LeadInfo:
    """Best-effort lead extraction from user text when model tags are missing."""
    updated = existing.model_dump()
    text = message.strip()

    if not updated.get("phone"):
        phone_match = re.search(r"(?:\+91[-\s]?)?(\d{10})\b", text)
        if phone_match:
            updated["phone"] = phone_match.group(1)

    if not updated.get("location"):
        location_match = re.search(r"\b(?:in|near|at|around)\s+([A-Z][a-zA-Z\s]+)\b", text)
        if location_match:
            candidate = location_match.group(1).strip()
            if len(candidate) > 3 and not any(bad in candidate.lower() for bad in ("under", "budget", "lakh", "crore")):
                updated["location"] = candidate

    if not updated.get("budget"):
        budget_match = re.search(
            r"\b(?:under|around|about|upto|up to|budget\s*(?:is|around|about)?\s*)?"
            r"((?:₹|rs\.?|inr)?\s*\d+(?:[\.,]\d+)?\s*(?:cr|crore|lakh|lac|million)?)\b",
            text,
            flags=re.IGNORECASE,
        )
        if budget_match:
            updated["budget"] = budget_match.group(1).strip()

    if not updated.get("name"):
        name_patterns = [
            r"\bmy name is\s+([A-Za-z][A-Za-z\s]{1,40})$",
            r"\bi am\s+([A-Za-z][A-Za-z\s]{1,40})$",
            r"\bthis is\s+([A-Za-z][A-Za-z\s]{1,40})$",
            r"\bcall me\s+([A-Za-z][A-Za-z\s]{1,40})$",
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, text, flags=re.IGNORECASE)
            if not name_match:
                continue
            candidate = name_match.group(1).strip(" .,")
            lower = candidate.lower()
            if any(bad in lower for bad in ("looking", "interested", "from", "for ")):
                continue
            words = [w for w in candidate.split() if w.isalpha()]
            if 1 <= len(words) <= 3:
                updated["name"] = " ".join(w.capitalize() for w in words)
                break

    return LeadInfo(**updated)


def extract_response_text(response) -> str:
    """Extract text from OpenAI-compatible chat completion responses safely."""
    try:
        content = response.choices[0].message.content
    except Exception:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts).strip()

    return ""


def prepare_chat_context(body: ChatRequest) -> tuple[list[dict], str, LeadInfo, Optional[dict]]:
    projects = load_projects()
    if not projects:
        raise HTTPException(status_code=400, detail="No projects found. Add at least one project first.")
    merged_lead = extract_lead_from_user_message(body.message, body.lead)
    resolved_language = resolve_chat_language(body.language, body.message, body.history)

    messages = [{"role": m.role, "content": m.content} for m in body.history]
    messages.append({"role": "user", "content": body.message})

    system = build_system_prompt(
        projects,
        resolved_language,
        merged_lead,
        body.voice_mode,
    )
    
    lang_name = LANGUAGES.get(resolved_language, LANGUAGES["en"])["label"]
    enforcement_suffix = f"\n\n[CRITICAL: You MUST reply STRICTLY in {lang_name} using ONLY the Latin/Romanized alphabet (e.g., Hinglish). Do NOT reply in English.]"

    payload_messages = [{"role": "system", "content": system}, *messages]
    if payload_messages[-1]["role"] == "user":
        payload_messages[-1]["content"] += enforcement_suffix

    return payload_messages, resolved_language, merged_lead, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return await render_index(request, admin_view=False)


@app.get("/admin", response_class=HTMLResponse)
async def admin_index(request: Request):
    return await render_index(request, admin_view=True)


async def render_index(request: Request, admin_view: bool):
    projects = load_projects()
    has_index_template = (TEMPLATES_DIR / "index.html").exists()
    if not has_index_template:
        count = len(projects)
        message = (
            "Templates folder is missing. API is running. "
            "Use /docs for API testing or add templates/index.html for UI."
        )
        return HTMLResponse(
            f"<html><body><h2>Propello AI API</h2><p>{message}</p><p>Projects loaded: {count}</p></body></html>"
        )
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "projects": projects,
            "languages": LANGUAGES,
            "admin_view": admin_view,
        },
    )


@app.post("/api/chat")
async def chat(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    payload_messages, resolved_language, merged_lead, _ = prepare_chat_context(body)

    client = get_llm_client()
    try:
        response = create_chat_completion_with_fallback(
            client=client,
            messages=payload_messages,
            max_tokens=450,
            stream=False,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
#     response = client.responses.create(
#     input="Explain the importance of fast language models",
#     model="",
# )
# # print(response.output_text)
    

    raw = extract_response_text(response)
    if not raw:
        raw = (
            "I am here with you. I can compare the best project options by budget, location, and configuration. "
            "Would you like me to start with options under your budget?"
        )
    clean, new_lead = parse_lead_tags(raw, merged_lead)
    new_lead = extract_lead_from_user_message(body.message, new_lead)
    persist_lead(new_lead, clean)

    return JSONResponse({
        "reply":           clean,
        "lead":            new_lead.model_dump(),
        "language":        resolved_language,
        "raw_for_history": raw,
    })


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    payload_messages, resolved_language, merged_lead, _ = prepare_chat_context(body)

    def event_stream():
        try:
            client = get_llm_client()
            stream = create_chat_completion_with_fallback(
                client=client,
                messages=payload_messages,
                max_tokens=450,
                stream=True,
            )

            all_text_parts: list[str] = []
            buffer = ""

            for chunk in stream:
                delta = ""
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""

                if not delta:
                    continue

                all_text_parts.append(delta)
                buffer += delta

                while True:
                    match = re.search(r"(.+?[.!?])(?:\s+|$)", buffer, flags=re.S)
                    if not match:
                        break
                    sentence = match.group(1).strip()
                    buffer = buffer[match.end():]
                    sentence_clean = re.sub(r"\[LEAD:[^\]]+\]", "", sentence).strip()
                    if sentence_clean:
                        payload = {"type": "chunk", "text": sentence_clean}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            raw = "".join(all_text_parts).strip()
            if not raw:
                raw = (
                    "I am here with you. I can compare the best project options by budget, location, and configuration. "
                    "Would you like me to start with options under your budget?"
                )

            trailing = re.sub(r"\[LEAD:[^\]]+\]", "", buffer).strip()
            if trailing:
                payload = {"type": "chunk", "text": trailing}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            clean, new_lead = parse_lead_tags(raw, merged_lead)
            new_lead = extract_lead_from_user_message(body.message, new_lead)
            persist_lead(new_lead, clean)

            done_payload = {
                "type": "done",
                "reply": clean,
                "lead": new_lead.model_dump(),
                "language": resolved_language,
                "raw_for_history": raw,
            }
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        except RuntimeError as exc:
            payload = {"type": "error", "detail": str(exc)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            payload = {"type": "error", "detail": str(exc)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/projects")
async def get_projects():
    return load_projects()


@app.get("/api/voice/capabilities")
async def voice_capabilities():
    return {
        "server_stt": has_llm_key(),
        "server_tts": has_elevenlabs_key(),
        "tts_provider": "elevenlabs" if has_elevenlabs_key() else "browser",
        "stt_provider": "groq_whisper" if has_llm_key() else "browser",
    }


@app.post("/api/voice/transcribe")
async def voice_transcribe(audio: UploadFile = File(...)):
    if not has_llm_key():
        raise HTTPException(status_code=501, detail="Server transcription unavailable. Set GROQ_API_KEY.")

    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        client = get_llm_client()
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
            )
        text = getattr(transcript, "text", "") or ""
        text = text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="Could not transcribe audio.")
        return {"text": text}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/api/voice/tts")
async def voice_tts(body: VoiceTTSRequest):
    if not has_elevenlabs_key():
        raise HTTPException(status_code=501, detail="Server TTS unavailable. Set ELEVENLABS_API_KEY.")
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    req = _build_elevenlabs_tts_request(text)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            audio_bytes = resp.read()
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=502, detail=f"ElevenLabs API error: {detail[:400]}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}")


@app.get("/api/voice/tts/stream")
async def voice_tts_stream(text: str, language: str = "en"):
    if not has_elevenlabs_key():
        raise HTTPException(status_code=501, detail="Server TTS unavailable. Set ELEVENLABS_API_KEY.")
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    def stream_audio():
        req = _build_elevenlabs_tts_request(text)
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    yield chunk
        except urllib.error.HTTPError as exc:
            # Streaming response already started; stop yielding on upstream failure.
            _ = exc.read()
            return
        except Exception:
            return

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")


@app.post("/api/projects/extract-source")
async def extract_project_source(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None)
):
    if not has_llm_key():
        raise HTTPException(status_code=501, detail="Server LLM unavailable. Set GROQ_API_KEY.")
    
    extracted_text = (raw_text or "").strip() + "\n\n"
    
    if file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in [".pdf", ".docx", ".txt"]:
            raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                doc = fitz.open(tmp_path)
                for page in doc:
                    extracted_text += page.get_text() + "\n"
                doc.close()
            elif suffix == ".docx":
                doc = docx.Document(tmp_path)
                for para in doc.paragraphs:
                    extracted_text += para.text + "\n"
            elif suffix == ".txt":
                with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text += f.read() + "\n"
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the provided source.")
        
    try:
        client = get_llm_client()
        system_prompt = """You are a highly capable real estate data extraction agent.
You will be provided with raw text from a real estate brochure or website.
Cleanly extract the project details strictly into the identical JSON schema below.
If info is missing, use empty strings or empty arrays. Do not guess.

JSON FORMAT TO STRICTLY FOLLOW:
{
  "name": "string",
  "developer": "string",
  "tagline": "string",
  "location": "string",
  "logo_emoji": "string (single building emoji)",
  "configs": "string (multiline, format: Type | Size | Price, e.g. 2 BHK | 950 sq ft | ₹2.5 Cr)",
  "highlights": "string (multiline, one highlight per line)",
  "amenities": "string (comma-separated, e.g. Swimming pool, Gym, Clubhouse)",
  "nearby": "string (multiline, format: Landmark | Distance, e.g. Metro Station | 5 min)"
}
OUTPUT ONLY STRICT RENDERABLE JSON WITHOUT ANY MARKDOWN BLOCKS AROUND IT."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract project specs from this text:\n\n{extracted_text[:20000]}"}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_json = response.choices[0].message.content
        extracted_data = json.loads(raw_json)
        return {"success": True, "data": extracted_data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM Extraction failed: {exc}")


@app.post("/api/lead/save")
async def save_lead_api(lead: LeadInfo):
    persist_lead(lead)
    return {"success": True}


@app.post("/api/projects/add")
async def add_project(body: AddProjectRequest):
    projects   = load_projects()
    project_id = (
        body.name.lower().replace(" ", "-")
        + "-"
        + body.developer.lower().replace(" ", "-")
    )

    if any(p["id"] == project_id for p in projects):
        raise HTTPException(status_code=409, detail="Project with this name/developer already exists")

    new_project = {
        "id":           project_id,
        "name":         body.name,
        "developer":    body.developer,
        "tagline":      body.tagline,
        "location":     body.location,
        "logo_emoji":   body.logo_emoji,
        "accent_color": body.accent_color,
        "configs":      [c.model_dump() for c in body.configs],
        "highlights":   body.highlights,
        "amenities":    body.amenities,
        "nearby":       [n.model_dump() for n in body.nearby],
        "connectivity": [n.model_dump() for n in body.connectivity],
        "payment_plan": body.payment_plan,
        "contact":      body.contact,
    }

    projects.append(new_project)
    save_projects(projects)
    return {"success": True, "project": new_project}


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    projects = load_projects()
    filtered = [p for p in projects if p["id"] != project_id]
    if len(filtered) == len(projects):
        raise HTTPException(status_code=404, detail="Project not found")
    save_projects(filtered)
    return {"success": True}


@app.get("/api/leads")
async def get_leads():
    """Return all captured leads from Google Sheets (if configured) or local CSV."""
    if _use_sheets():
        try:
            ws = _leads_sheet()
            records = ws.get_all_records()
            return records
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Sheets read error: {e}")
    # Fallback: local CSV
    leads = []
    if LEADS_FILE.exists():
        try:
            with open(LEADS_FILE, "r", encoding="utf-8") as f:
                leads = list(csv.DictReader(f))
        except Exception:
            pass
    return leads


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
