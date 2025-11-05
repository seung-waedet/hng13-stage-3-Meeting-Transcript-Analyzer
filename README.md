# 🤖 Meeting Transcript Analyzer Agent

> An AI-powered agent that analyzes meeting transcripts and extracts actionable insights using Google's Gemini 2.0 Flash model.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-orange.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built with FastAPI and integrated with Telex.im using the A2A protocol.

---

## 🚀 Quick Start

**New here?** → Read **[START_HERE.md](START_HERE.md)** for a complete setup guide (10 minutes)

**Want A2A details?** → Read **[A2A_PROTOCOL.md](A2A_PROTOCOL.md)** for full protocol implementation

**Already setup?** → Jump to [Usage](#usage) or visit http://localhost:8000/docs

---

## ⭐ A2A Protocol Compliance

This agent **strictly follows the A2A (Agent-to-Agent) Protocol** specification:

- ✅ **JSON-RPC 2.0** format for all requests/responses
- ✅ **Proper message structure** with `kind`, `role`, and `parts`
- ✅ **Task lifecycle management** (`working`, `completed`, `input-required`, `failed`)
- ✅ **Structured artifacts** for machine-readable outputs
- ✅ **Context management** for multi-turn conversations
- ✅ **Webhook support** for asynchronous processing
- ✅ **Error handling** with JSON-RPC error codes

**Primary endpoint:** `POST /a2a/analyze`

See [A2A_PROTOCOL.md](A2A_PROTOCOL.md) for complete implementation details.

---

## Features

- 📝 **Smart Summarization**: Generates concise meeting summaries
- ✅ **Action Item Extraction**: Identifies tasks and assigns ownership
- 🎯 **Decision Tracking**: Captures key decisions made during meetings
- 👥 **Participant Recognition**: Identifies meeting participants
- ➡️ **Next Steps**: Outlines follow-up actions
- 🔗 **A2A Protocol Compliant**: Full JSON-RPC 2.0 implementation
- 🤝 **Telex.im Integration**: Seamless platform communication
- 📦 **Structured Artifacts**: Rich, machine-readable outputs
- 💬 **Context Management**: Multi-turn conversation support

## Prerequisites

- Python 3.8+
- Google Gemini API key (get it from https://makersuite.google.com/app/apikey)
- Telex.im channel access

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd meeting-transcript-analyzer
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```
GEMINI_API_KEY=your_gemini_api_key_here
TELEX_CHANNEL_ID=your_channel_id_here
AGENT_PORT=8000
```

## Running the Agent

### Local Development

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The agent will be available at `http://localhost:8000`

### Interactive API Documentation

Once running, access the interactive Swagger UI:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive testing of all endpoints with example requests and responses.

### API Endpoints

- `GET /` - Agent information
- `GET /health` - Health check
- `POST /a2a/analyze` - **A2A Protocol endpoint (JSON-RPC 2.0)** ⭐
- `GET /docs` - Interactive Swagger UI documentation
- `GET /redoc` - Alternative ReDoc documentation

**Main endpoint for Telex.im integration:** `/a2a/analyze`

## Usage

### Option 1: Interactive Swagger UI (Recommended)

1. Start the agent: `python main.py`
2. Open your browser: http://localhost:8000/docs
3. Click on the `/analyze` endpoint
4. Click "Try it out"
5. Paste your meeting transcript in the request body
6. Click "Execute"
7. See the results instantly!

The Swagger UI provides:
- Interactive testing of all endpoints
- Example requests and responses
- Request/response schema validation
- Easy debugging

### Option 2: Direct API Call (A2A Protocol)

```bash
curl -X POST http://localhost:8000/a2a/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "req-001",
    "method": "message/send",
    "params": {
      "message": {
        "kind": "message",
        "role": "user",
        "parts": [
          {
            "kind": "text",
            "text": "Meeting started at 10am. John discussed Q4 targets. Sarah agreed to prepare the report by Friday. We decided to increase marketing budget by 20%. Next meeting scheduled for next Monday."
          }
        ]
      },
      "configuration": {
        "blocking": true
      }
    }
  }'
```

### Telex.im Integration

1. Get your Telex organization access:
```bash
/telex-invite your-email@example.com
```

2. Deploy your agent to a public endpoint (use ngrok for testing):
```bash
ngrok http 8000
```

3. Configure Telex.im webhook to point to:
```
https://your-domain.com/webhook/telex
```

4. Send meeting transcripts to your Telex channel and the agent will respond with analysis

### Example Transcript

Send this to the agent:
```
Team standup meeting - Nov 5, 2025

John: Good morning everyone. Let's start with updates.

Sarah: I completed the user authentication module. It's ready for review.

Mike: I'm working on the payment integration. Should be done by Thursday.

John: Great. Sarah, can you review Mike's code once it's ready?

Sarah: Sure, I'll do that on Friday.

John: We need to decide on the database migration strategy.

Team: After discussion, we agreed to use blue-green deployment.

John: Perfect. Next meeting is Monday at 10am. Meeting adjourned.
```

## Response Format

The agent returns structured analysis:

```json
{
  "summary": "Team discussed project updates including completed authentication module and ongoing payment integration work.",
  "action_items": [
    "Sarah to review Mike's payment integration code on Friday",
    "Mike to complete payment integration by Thursday"
  ],
  "key_decisions": [
    "Use blue-green deployment for database migration"
  ],
  "participants_mentioned": ["John", "Sarah", "Mike"],
  "next_steps": [
    "Next meeting scheduled for Monday at 10am"
  ]
}
```

## Deployment

### Using Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t meeting-analyzer .
docker run -p 8000:8000 --env-file .env meeting-analyzer
```

### Cloud Deployment

Deploy to platforms like:
- Railway
- Render
- Heroku
- AWS Lambda
- Google Cloud Run

## Testing

### Option 1: Automated Test Script

```bash
python test_agent.py
```

### Option 2: Swagger UI

1. Visit http://localhost:8000/docs
2. Test each endpoint interactively
3. See real-time responses

### Option 3: Manual cURL

Test the health endpoint:
```bash
curl http://localhost:8000/health
```

Test A2A analysis:
```bash
curl -X POST http://localhost:8000/a2a/analyze \
  -H "Content-Type: application/json" \
  -d @sample_transcript.json
```

## Error Handling

The agent handles:
- Short/invalid transcripts (minimum 50 characters)
- Gemini API failures
- JSON parsing from LLM responses (handles markdown code blocks)
- Telex.im communication errors
- Malformed requests

## Architecture

```
┌─────────────┐
│  Telex.im   │
│   Channel   │
└──────┬──────┘
       │
       │ Webhook
       ▼
┌─────────────────┐
│   FastAPI       │
│   Agent         │
│  /webhook/telex │
│  + Swagger Docs │
└────────┬────────┘
         │
         │ Process
         ▼
┌─────────────────┐
│ Gemini 2.0 Flash│
│  (Google AI)    │
└────────┬────────┘
         │
         │ Analysis
         ▼
┌─────────────────┐
│   Response      │
│   Formatter     │
└────────┬────────┘
         │
         │ Send back
         ▼
┌─────────────────┐
│   Telex.im      │
│   Channel       │
└─────────────────┘
```

## Monitoring

View agent interactions:
```
https://api.telex.im/agent-logs/{channel-id}.txt
```

## 📁 Project Structure

```
meeting-transcript-analyzer/
├── main.py                    # FastAPI application
├── test_agent.py              # Automated tests
├── requirements.txt           # Dependencies
├── setup.sh                   # Setup script
├── START_HERE.md              # Setup guide
├── README.md                  # This file
└── [10+ documentation files]  # Comprehensive guides
```

See `PROJECT_STRUCTURE.txt` for complete file tree.

## 📖 Documentation

| File | Purpose |
|------|---------|
| [START_HERE.md](START_HERE.md) | Complete setup guide (start here!) |
| [A2A_PROTOCOL.md](A2A_PROTOCOL.md) | **A2A Protocol implementation details** |
| [QUICKSTART.md](QUICKSTART.md) | Fast setup alternative |
| [SWAGGER_GUIDE.md](SWAGGER_GUIDE.md) | Interactive API testing |
| [API_EXAMPLES.md](API_EXAMPLES.md) | Code examples |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment guide |
| [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) | HNG submission |

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License

## 💬 Support

For issues or questions:
- Check the documentation files
- Open an issue on GitHub
- Contact via Telex.im

## 🏆 Acknowledgments

- Built for HNG Internship Stage 3 Backend Task
- Powered by Google Gemini 2.0 Flash
- Integrated with Telex.im

## 🔗 Links

- **Gemini API**: https://ai.google.dev/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Telex.im**: https://telex.im/
- **HNG Internship**: https://hng.tech/

---

**Made with ❤️ for HNG Internship Stage 3**
