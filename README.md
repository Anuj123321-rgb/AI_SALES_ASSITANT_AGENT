# Social-to-Lead Agentic Workflow (AutoStream)

This repository contains an assignment implementation for ServiceHive's **Inflx** use case.
It builds a conversational AI agent for fictional SaaS company **AutoStream**.

## Features Implemented

- **Intent identification**
  - Casual greeting
  - Product/pricing inquiry
  - High-intent lead
- **RAG over local knowledge base** (`knowledge_base.md`)
  - Basic and Pro pricing/features
  - Company policy retrieval
- **Lead capture workflow with guardrails**
  - Collects Name, Email, Creator Platform
  - Calls `mock_lead_capture(name, email, platform)` **only after all are collected**
- **State management across turns** using **LangGraph** state.

## Project Structure

- `main.py` — CLI entrypoint
- `src/agent.py` — LangGraph agent, intent detection, RAG, and tool logic
- `knowledge_base.md` — local source data for retrieval
- `requirements.txt` — dependencies

## Setup & Run

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure API key

This implementation uses `gpt-4o-mini` via LangChain OpenAI wrapper:

```bash
export OPENAI_API_KEY="your_key_here"
```

### 3) Run

```bash
python main.py
```

## Example Conversation

1. User: `Hi, tell me about your pricing.`
2. Agent returns Basic/Pro details from local knowledge base.
3. User: `I want to try the Pro plan for my YouTube channel.`
4. Agent asks missing lead fields.
5. User provides name + email + platform.
6. Agent triggers:
   ```python
   mock_lead_capture(name, email, platform)
   ```

---

## Architecture Explanation (~200 words)

I chose **LangGraph** because the task requires reliable multi-turn behavior with explicit state and predictable tool execution. A plain chat loop can work for simple Q&A, but this assignment needs controlled transitions (intent → retrieval or lead qualification → tool call) and memory across 5–6 turns. LangGraph provides a strong foundation for this by representing state as a typed object and executing a deterministic node pipeline.

The state (`AgentState`) stores conversation messages, latest intent label, collected lead fields (`name`, `email`, `platform`), and a `lead_captured` flag. On every turn, the agent updates state first, then decides response behavior. Intent is detected using a hybrid strategy: lightweight keyword checks for high-intent phrases plus an LLM classifier prompt for ambiguous cases. For product inquiries, a local RAG flow uses BM25 retrieval against `knowledge_base.md`, then prompts the LLM to answer strictly from retrieved context.

For lead capture safety, the agent extracts structured fields from user text and calls `mock_lead_capture` only when all required values exist. If any field is missing, it asks specifically for pending details. This creates a deployable pattern: retrieval-augmented responses combined with guarded action execution.

## WhatsApp Deployment via Webhooks (Design)

To integrate with WhatsApp, I would use the **WhatsApp Business Cloud API** with a webhook endpoint:

1. **Webhook receiver** (FastAPI/Flask)
   - Verify Meta webhook challenge.
   - Receive incoming message events.
2. **Session store**
   - Map `wa_user_id -> AgentState` in Redis/Postgres.
   - Load state for each incoming message, run `agent.chat()`, then persist updated state.
3. **Reply sender**
   - Send agent response using Graph API `/messages` endpoint.
4. **Lead handoff**
   - When `mock_lead_capture` succeeds, replace mock with CRM API call.
5. **Reliability & security**
   - Validate webhook signatures.
   - Add retries/idempotency keys to avoid duplicate lead creation.
   - Log intents, tool calls, and failures for analytics.

This architecture keeps channel transport (WhatsApp) separate from agent reasoning, making the core workflow reusable across web chat, Instagram, or other channels.
