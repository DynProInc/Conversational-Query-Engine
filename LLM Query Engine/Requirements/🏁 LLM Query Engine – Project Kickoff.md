## 🏁 **LLM Query Engine – Project Kickoff**

------

### 🎯 **Why We’re Here**

We’re building a **Conversational Query Engine** to empower business users to ask data questions in plain English — and get real answers from our Snowflake Gold Layer.

This changes how teams interact with data:
 ⚙️ No SQL skills needed
 📣 Just ask → get answers → act faster

------

### 🧭 **Project Goals**

- Let users **speak or type** natural language questions
- **Translate** them into safe, accurate SQL
- **Execute** on Snowflake Gold Layer
- Return results in a clean, responsive UI
- **Capture feedback** to improve over time

------

### 🛠️ **System Overview**

```plaintext
[Voice/Text Input]
       ↓
[Prompt Engine + Metadata Injection]
       ↓
[LLM → SQL]
       ↓
[SQL Validator + Execution (Snowflake)]
       ↓
[Response + Feedback Capture]
```

Each layer is modular — and we're parallelizing across 5 engineers.

------

## 👷 **Team Breakdown – 5 Parallel Tracks**

------

### 1️⃣ **Engineer 1 – Voice to Prompt Capture**

🎤 **What You'll Build**:

- Voice input (e.g., ElevenLabs or browser native)
- Reliable transcription pipeline
- Submit prompt to backend
- Basic UI to test live voice queries

📦 Output: JSON with prompt + timestamp + user metadata

------

### 2️⃣ **Engineer 2 – Metadata Catalog + Mappings**

📚 **What You'll Build**:

- Extract table/column metadata from Snowflake
- Map business-friendly terms (e.g., “Customer Region”) to fields
- Structure in JSON/Firebase schema
- Serve mappings to LLM engine

📦 Output: JSON metadata store + lookup API/module

------

### 3️⃣ **Engineer 3 – Prompt Engineering + LLM API**

🧠 **What You'll Build**:

- Prompt templates: few-shot, contextual, with metadata
- LLM wrapper using OpenAI/BEDROCK/OSS
- Inject metadata into prompt dynamically
- Log prompt → SQL mappings

📦 Output: `/generateSQL(prompt, metadata)`

------

### 4️⃣ **Engineer 4 – SQL Runner + Guardrails**

🔐 **What You'll Build**:

- Validate generated SQL (block DML, timeouts, etc.)
- Execute query on Snowflake (read-only)
- Handle pagination, performance limits
- Return formatted results (tabular JSON)

📦 Output: `/runQuery(sql) → results + metadata`

------

### 5️⃣ **Engineer 5 – Feedback + Admin Panel**

🔁 **What You'll Build**:

- Capture user feedback on results (1–5 stars + notes)
- Store in Firebase/Snowflake for review
- Admin UI to view low-rated queries
- Prep for model fine-tuning phase

📦 Output: `/submitFeedback()` + admin dashboard

------

### 🗓️ **Sprint 1 Plan (2 Weeks)**

| Day      | Goal                                                    |
| -------- | ------------------------------------------------------- |
| Day 1    | Kickoff + repo setup                                    |
| Day 2–6  | Each engineer builds a self-contained working prototype |
| Day 7    | Mid-sprint integration sync                             |
| Day 8–13 | Connect modules (prompt → SQL → execution)              |
| Day 14   | Demo end-to-end flow + collect feedback                 |

------

### ✅ **Definition of Done (Phase 1 MVP)**

- Voice/text input working
- Prompt → SQL conversion functional for 5 core questions
- Safe, successful execution on Snowflake
- Results returned in clean JSON
- Feedback capture mechanism live
- All logs flow to Firebase or logging store

------

### 🔁 **Working Style**

- Daily async check-ins (Slack/Notion updates)
- Weekly 30-min team sync
- Code-first, shipping mindset
- Shared repo, clear commit history
- Modular ownership, unblock fast

------

### 📦 **Tech Stack**

| Layer       | Tooling                                       |
| ----------- | --------------------------------------------- |
| Voice Input | ElevenLabs / Web Speech API                   |
| LLM         | OpenAI / Bedrock / OSS                        |
| Metadata    | Snowflake + Firebase                          |
| Execution   | Snowflake (read-only)                         |
| UI          | React (Shadcn UI, Tailwind), Firebase Hosting |
| Feedback    | Firebase / Snowflake                          |
| Logs        | Firestore / Console / Sentry (optional)       |

------

### 🎉 **Let’s Build This Right**

This is foundational. If we build it cleanly, it becomes the pattern for future LLM products.

✅ Start lean
 ✅ Integrate weekly
 ✅ Ship working slices
 ✅ Collect real feedback early