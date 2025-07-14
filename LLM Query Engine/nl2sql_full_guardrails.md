# 🧠 Natural Language to SQL (NL2SQL) – Full Strategy with Guardrails

A complete, production-grade blueprint for building safe, accurate, and scalable NL2SQL systems using LLMs. This document is designed for junior to mid-level developers and data engineers.

---

## 🔧 A. Prompt-Level Guardrails

| Type               | Guardrail                                                                 | Purpose                             |
|--------------------|--------------------------------------------------------------------------|-------------------------------------|
| **Scope**          | Only allow `SELECT` statements                                            | Prevent data mutation               |
| **Field control**  | Use only explicitly mentioned fields                                       | Prevent hallucination               |
| **Join control**   | Only allow joins defined in schema metadata                               | Prevent invalid or expensive joins  |
| **Derived metrics**| Only include when explicitly requested                                     | Avoid overreach                     |
| **Ambiguity**      | Return comment if unclear (e.g., `-- Please clarify metric`)              | Force precision                     |
| **Row limits**     | Add `LIMIT 100` unless aggregate query                                    | Cost and performance safety         |
| **Date handling**  | Use standardized logic (`CURRENT_DATE`, ISO)                              | Consistency                         |
| **Formatting**     | Return **only SQL**, no explanations                                      | Parse-ready output                  |

---

## 📊 B. Schema & Semantic Layer Guardrails

| Component               | Guardrail                                                    | Tooling/Notes                     |
|-------------------------|--------------------------------------------------------------|----------------------------------|
| Schema registry         | Centralized schema definition                                | dbt, DataHub, or custom YAML     |
| Semantic layer          | Formal definitions for metrics like `net_sales`              | dbt metrics, Cube.dev            |
| Join path control       | Whitelisted relationships                                    | Graph-based join paths           |
| Field typing            | Block invalid aggregations (e.g., `SUM` on text)             | SQL parser or AST-based rules    |
| Field-level access      | Restrict sensitive fields (e.g., salary, PII)                 | RBAC/ABAC policy integration     |

---

## 🧪 C. Output Validator Layer (Post-LLM Safety)

| Check                           | Tool/Method        |
|----------------------------------|--------------------|
| SQL syntax valid                | `sqlglot`, `sqlparse` |
| All columns exist in schema     | Schema validator   |
| Aggregations used correctly     | Semantic rules     |
| No `SELECT *`                   | AST enforcement    |
| Function whitelist              | Regex or AST       |
| Estimated cost check            | Use `EXPLAIN`      |

---

## 🧠 D. Feedback Loop + Human-in-the-Loop (HITL)

| Feature                      | Purpose                                      |
|-----------------------------|----------------------------------------------|
| SQL preview                 | Prevents unintended queries                  |
| Editable query block        | User can tweak generated SQL                 |
| Feedback rating             | Quality training data                        |
| Versioning                  | Track evolution of prompts & outputs         |
| Correction suggestion       | Enables improvement of prompt design         |

---

## 📜 E. Governance & Observability

| What to log                         | Why it matters                     |
|-------------------------------------|------------------------------------|
| Prompt + schema + user query        | Full context for reproducibility   |
| Generated SQL                       | Auditing and debugging             |
| Execution metadata (cost, time)     | Cost visibility, optimization      |
| Access logs                         | Security, compliance (GDPR, SOC2)  |
| Feedback logs                       | Continuous improvement             |

---

## 🔐 F. Security & Access Controls

| Area                     | Guardrail                           |
|--------------------------|-------------------------------------|
| Authentication           | Firebase Auth or OAuth              |
| Column-level ACL         | Block access to fields per role     |
| Row-level security       | Apply user-specific filters         |
| Prompt filtering         | Remove illegal table/field refs     |
| Token leakage prevention | Prevent injection of private logic  |

---

## 💸 G. Query Cost Management

| Strategy                     | Benefit                        |
|-----------------------------|--------------------------------|
| Add `LIMIT` to all queries  | Cost and performance control   |
| Show estimated cost         | Snowflake `EXPLAIN` or logic   |
| Timeout/abort long queries  | Prevent runaways               |
| Sample large datasets       | Fast prototyping               |
| Rate limiting               | Avoid abuse or spikes          |

---

## 🧩 H. Developer/Tooling Integration

| Integration            | Purpose                                |
|------------------------|----------------------------------------|
| Firebase Hosting       | Secure web-based LLM frontend          |
| Firestore Backend      | Store prompt templates + feedback      |
| Type-safe schema sync  | Keep prompt schema in sync with DB     |
| RAG (e.g., embeddings) | Prevent hallucination via retrieval    |
| VSCode plugin / CLI    | Local developer testing & debugging    |

---

## 🧱 I. System Architecture Overview

```
User → UI (React + Firebase)  
     → Prompt Builder (Injects schema + user query)  
     → LLM (e.g., GPT-4 or Claude)  
     → SQL Validator (e.g., sqlglot, business rules)  
     → Human-in-the-loop Preview (optional)  
     → Query Executor (Snowflake)  
     → Logging & Feedback Capture  
```

---

## 🧰 J. Reusable Prompt Template

```
You are a SQL generation assistant. Convert the natural language question below into a Snowflake SQL SELECT query.

Schema:
{{schema metadata: tables, columns, joins}}

Business Definitions:
{{e.g. net_sales = sales_amount - discounts}}

Rules:
- Only generate SELECT queries.
- Use only fields and metrics mentioned in the query.
- Do not include derived metrics unless specifically requested.
- Join only as specified in metadata.
- Add LIMIT 100 unless aggregation.
- If unclear, respond with:
  -- Unable to generate query. Please clarify.

User Query:
"{{natural language query}}"

Output:
Only the SQL query. No explanation.
```

---

## 📘 K. Advanced Enhancements (Optional but Valuable)

| Feature                          | Description                                        |
|----------------------------------|----------------------------------------------------|
| Synonym Mapping                  | "Revenue" → `net_sales` using controlled vocab     |
| Query Disambiguation             | Ask follow-ups if query is vague                  |
| Multilingual Support             | Translate natural language queries                 |
| Fallback Model Tier              | Switch to RAG/template/fine-tuned fallback         |
| User Personalization             | Save query history, reuse past metrics             |
| Prompt & SQL Testing Framework   | Unit tests + golden queries for regression control |
| Immutable Query Logging          | For compliance (GDPR, SOC2, HIPAA, etc.)           |

---

## ✅ Final Checklist for Production Readiness

- [x] Prompt structured with guardrails
- [x] Schema metadata dynamically injected
- [x] Semantic layer enforced (via dbt or metadata model)
- [x] SQL validation layer present
- [x] Audit logging enabled
- [x] UI/UX for query preview and editing
- [x] Feedback capture system in place
- [x] Rate limiting + query cost control
- [x] Access controls (field/table/row)
- [x] Human-in-the-loop review (optional)

---

This document is designed to be shared across engineering, analytics, and data science teams for consistent NL2SQL implementation and safety.
