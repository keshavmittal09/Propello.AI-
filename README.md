<h1 align="center">
  🏠 Propello AI
</h1>

<p align="center">
  <strong>The World's Smartest Real Estate Sales Agent — Powered by AI, Driven by Results.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Groq-llama--3.3--70b-F26522?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Vapi.ai-Voice%20Agent-6C47FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Languages-10%20Indian-FF6B35?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

---

> **Propello AI is not just a chatbot. It's a full-stack, AI-powered real estate sales platform** that converses with buyers in text _and_ voice, captures leads automatically, schedules appointments, prevents booking conflicts, and stores everything in a persistent CRM — all without a single human agent being online.

---

## ✨ Why Propello AI?

The Indian real estate market is booming — but most developers are still relying on cold calls and static brochures to convert leads. **Propello AI changes everything.**

- 🚀 **Instant 24/7 response** — Your AI advisor "Priya" never sleeps, never misses a lead.
- 🌐 **10 Indian languages** — Reach buyers in Hindi, Gujarati, Tamil, Marathi, and more.
- 📞 **Voice + Chat** — Buyers can call Priya AND chat with her in the same session.
- 📊 **Auto-CRM** — Every lead, budget, and appointment is silently captured to a CSV.
- 🏗️ **Zero-code project management** — Add new properties from a PDF, Word doc, or raw text.

---

## 🎯 Core Features

### 🤖 AI Sales Advisor "Priya"
- Powered by **Groq's `llama-3.3-70b-versatile`** — one of the fastest open-weight LLMs available.
- Deep contextual understanding of every property in your portfolio.
- Trained to qualify leads, handle objections, and close site-visit bookings.
- Responds naturally without ever revealing she is an AI.

### 📞 Real-Time Voice Calling (Vapi.ai)
- **WebRTC-based, browser-native voice calls** — zero app download required.
- Powered by **GPT-4o-mini** for ultra-low latency responses during the call.
- Speak to Priya in **any Indian language** and she replies in the same language.
- **Barge-in interruption support** — Priya stops talking the instant you speak.
- Azure Neural voices (`hi-IN-SwaraNeural`, `en-IN-NeerjaNeural`, etc.) for a natural, human-like sound.

### 🌐 10-Language Multilingual Support
| Language | Code | Voice |
|----------|------|-------|
| English (India) | `en` | en-IN-NeerjaNeural |
| Hindi | `hi` | hi-IN-SwaraNeural |
| Marathi | `mr` | mr-IN-AarohiNeural |
| Gujarati | `gu` | gu-IN-DhwaniNeural |
| Tamil | `ta` | ta-IN-PallaviNeural |
| Telugu | `te` | te-IN-ShrutiNeural |
| Bengali | `bn` | bn-IN-TanishaaNeural |
| Kannada | `kn` | kn-IN-SapnaNeural |
| Malayalam | `ml` | ml-IN-SobhanaNeural |
| Punjabi | `pa` | en-IN-NeerjaNeural |

### 📋 Automatic Lead Capture & CRM
- Lead details (Name, Phone, Budget, Location) are silently extracted from **both chat and voice calls**.
- Voice leads use **OpenAI Function Calling** — no text parsing, no data loss.
- All data is written to `data/leads.csv` in real-time.
- Intelligent deduplication — updates existing records instead of creating duplicates.
- **Appointment scheduling** with automatic **collision detection** — no double bookings, ever.

### 🏗️ Magic Project Ingestion (No-Code)
- Add new properties by **pasting a brochure, uploading a PDF, or uploading a Word document**.
- An LLM pipeline automatically extracts and structures: name, developer, configurations, amenities, highlights, nearby landmarks, and payment plan.
- Projects are immediately available to Priya in all future voice and chat sessions.

### 📱 Premium Chat UI
- Glassmorphism design with micro-animations and a clean, modern aesthetic.
- Real-time **lead strip** that updates as Priya collects buyer details.
- Contextual **quick-reply chips** to guide the conversation.
- Live call overlay with animated sound wave and call timer.

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PROPELLO AI                          │
│                                                         │
│   Browser (HTML/JS)                                     │
│   ├── Chat Interface (text → /api/chat)                 │
│   └── Voice Interface (WebRTC → Vapi.ai)                │
│                           │                             │
│   FastAPI Backend (app.py)                              │
│   ├── /api/chat           → Groq LLM (llama-3.3-70b)   │
│   ├── /api/lead/save      → data/leads.csv              │
│   ├── /api/projects/*     → data/projects.json          │
│   └── /api/projects/extract-source → LLM Auto-Fill      │
│                                                         │
│   Storage                                               │
│   ├── data/projects.json  (Property database)           │
│   └── data/leads.csv      (CRM / Lead database)         │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/propello-ai.git
cd propello-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=gsk_...          # Get from console.groq.com (free)
VAPI_PUBLIC_KEY=...            # Get from vapi.ai/dashboard → API Keys
ELEVEN_API_KEY=sk_...          # Optional: for ElevenLabs TTS
```

> **Note:** The app reads `.env` automatically — no `export` commands needed!

### 4. Run the server
```bash
python app.py
# or
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser. That's it. 🚀

---

## ☁️ Deploy to Render (1-Click)

This repository includes a `render.yaml` Blueprint for automatic cloud deployment.

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New +** → **Blueprint**.
3. Connect your repo. Render reads `render.yaml` and auto-configures:
   - Python environment & startup command
   - 1 GB persistent disk for `data/` (leads + projects survive restarts)
4. Add your **3 environment variables** in the Render dashboard.
5. Hit **Deploy**. Your live URL will be ready in ~2 minutes.

---

## 📁 Project Structure

```
propello-ai/
├── app.py                    # FastAPI backend — all API routes & logic
├── requirements.txt          # Python dependencies
├── render.yaml               # Render.com deployment blueprint
├── Procfile                  # Heroku / Railway startup command
├── .env                      # API keys (never commit this!)
├── .gitignore                # Protects .env and leads.csv from GitHub
├── data/
│   ├── projects.json         # 🏠 Property database (persists across restarts)
│   └── leads.csv             # 📊 Live CRM — every lead captured here
└── templates/
    └── index.html            # Full chat + voice UI (single-file frontend)
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main chat interface |
| `POST` | `/api/chat` | Send message, get AI reply with lead extraction |
| `GET` | `/api/projects` | List all properties |
| `POST` | `/api/projects/add` | Manually add a property |
| `DELETE` | `/api/projects/{id}` | Remove a property |
| `POST` | `/api/projects/extract-source` | Auto-fill from PDF / DOCX / text |
| `POST` | `/api/lead/save` | Save / update lead in CRM (used by voice agent) |
| `GET` | `/api/leads` | Fetch all captured leads |

---

## 🛡️ Security

- All API keys are loaded from `.env` — never hard-coded.
- `.gitignore` ensures keys and private lead data are never pushed to GitHub.
- Appointment collision detection prevents any two buyers from booking the same slot.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.10+ |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Voice AI | Vapi.ai + GPT-4o-mini |
| Speech-to-Text | Deepgram (`nova-2` / multilingual) |
| Text-to-Speech | Microsoft Azure Neural Voice |
| PDF Parsing | PyMuPDF (fitz) |
| DOCX Parsing | python-docx |
| Storage | JSON (projects) + CSV (leads) |
| Frontend | Vanilla HTML5, CSS3, JavaScript ES6 |
| Deployment | Render.com (with persistent disk) |

---

## 📬 Contact & Credits

Built with ❤️ by **Keshav Mittal**

> *Propello AI — Because every serious buyer deserves an advisor who never sleeps.*
