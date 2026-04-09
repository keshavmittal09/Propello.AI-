import os
import re
import json
import urllib.request
import urllib.error
import tempfile
import fitz
import docx
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from openai import OpenAI

try:
    import gspread
    from google.oauth2.service_account import Credentials as GCredentials
    _GSPREAD_AVAILABLE = True
except ImportError:
    _GSPREAD_AVAILABLE = False


app = FastAPI(title="Propello AI", version="1.0.0")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
ENV_FILE = BASE_DIR / ".env"


def load_local_env():
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_local_env()


def get_llm_client():
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(500, "Missing API key")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


class LeadInfo(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    budget: Optional[str] = None
    location: Optional[str] = None
    appointment: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    lead: LeadInfo = LeadInfo()
    language: str = "auto"
    voice_mode: bool = False


import csv
from datetime import datetime

LEADS_FILE = DATA_DIR / "leads.csv"


def log_lead_to_csv(lead: LeadInfo):
    if not lead.phone:
        return

    DATA_DIR.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    if LEADS_FILE.exists():
        with open(LEADS_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    rows.append({
        "Timestamp": now,
        "Name": lead.name,
        "Phone": lead.phone,
        "Budget": lead.budget,
        "Location": lead.location,
        "Appointment": lead.appointment
    })

    with open(LEADS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp","Name","Phone","Budget","Location","Appointment"])
        writer.writeheader()
        writer.writerows(rows)


def log_lead_to_crm(lead: LeadInfo, transcript=""):
    if not lead.phone:
        return

    payload = {
        "name": lead.name,
        "phone": lead.phone,
        "budget": lead.budget,
        "location": lead.location,
        "appointment": lead.appointment,
        "transcript": transcript
    }

    url = os.environ.get("CRM_WEBHOOK_URL")
    if not url:
        return

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception as e:
        print("CRM Error:", e)


def get_memory(phone):
    if not phone:
        return None
    try:
        url = f"{os.environ.get('CRM_LOOKUP_URL')}/{phone}"
        with urllib.request.urlopen(url, timeout=2) as r:
            data = json.loads(r.read().decode())
            return data.get("priya_memory_brief")
    except:
        return None


def build_system_prompt(language, lead, memory=None):
    known = []

    if memory:
        known.append(f"Past: {memory}")

    if lead.name:
        known.append(f"Name: {lead.name}")
    if lead.phone:
        known.append("Phone known")
    if lead.budget:
        known.append(f"Budget: {lead.budget}")
    if lead.location:
        known.append(f"Location: {lead.location}")

    return f"\nYou are a real estate sales assistant.\n\nKnown:\n{chr(10).join(known)}\n\nBe short, human, and always move toward booking.\n"


@app.post("/api/chat")
async def chat(body: ChatRequest):

    lead = body.lead

    match = re.search(r"\d{10}", body.message)
    if match:
        lead.phone = match.group()

    memory = get_memory(lead.phone)

    system = build_system_prompt(body.language, lead, memory)

    client = get_llm_client()

    messages = [{"role": "system", "content": system}]
    messages += [{"role": m.role, "content": m.content} for m in body.history]
    messages.append({"role": "user", "content": body.message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    reply = response.choices[0].message.content

    log_lead_to_csv(lead)
    log_lead_to_crm(lead, reply)

    return {
        "reply": reply,
        "lead": lead.model_dump()
    }
